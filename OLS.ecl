/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems.  All rights reserved.
############################################################################## */

IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT PBblas;
IMPORT PBblas.Types as pbbTypes;
IMPORT PBblas.Converted as pbbConverted;
IMPORT PBblas.MatUtils;
IMPORT ML_Core.Math as Math;

NotCompat := PBblas.Constants.Dimension_Incompat;
NumericField := Types.NumericField;
Layout_Cell := pbbTypes.Layout_Cell;
Layout_Model := Types.Layout_Model;
t_work_item  := Types.t_work_item;
t_RecordID   := Types.t_RecordID;
t_FieldNumber := Types.t_FieldNumber;
t_FieldReal   := Types.t_FieldReal;
t_Discrete    := Types.t_Discrete;
t_Count       := Types.t_Count;
empty_data := DATASET([], NumericField);
triangle := pbbTypes.triangle;
side := pbbTypes.side;
diagonal := pbbTypes.diagonal;
/**
  * Ordinary Least Squares (OLS) Linear Regression
  *  aka Ordinary Linear Regression.
  *
  * <p>Regression learns a function that maps a set of input data (independents)
  * to one or more output variables (dependents).  The resulting learned function is
  * known as the model.  That model can then be used repetitively to predict
  * (i.e. estimate) the output value(s) based on new input data.
  *
  * Two major use cases are supported:<ol>
  * <li>Learn and return a model.</li>
  * <li>Use an existing (e.g. persisted) model to predict new values for Y.</li></ol>
  *
  * <p>Of course, both can be done in a single run.  Alternatively, the
  * model can be persisted and used indefinitely for prediction of Y values,
  * as long as the record format has not changed, and the original training
  * data remains representative of the population.
  *
  * <p>OLS supports any number of independent variables (Multiple Regression) and
  * multiple dependent variables (Multivariate Regression).  In this way, multiple
  * variables' values can be predicted from the same input (i.e. independent)
  * data.
  *
  * <p>Training data is presented as parameters to this module.  When using a
  * previously persisted model (use case 2 above), these parameters should
  * be omitted.
  *
  * <p>This module provides a rich set of analytics to assess the usefulness of the
  * resulting linear regression model, and to determine the best subset of
  * independent variables to include in the model.  These include:<ul>
  *
  * <li>For the whole model:<ul>
  *   <li>Analysis of Variance (ANOVA)</li>
  *   <li>R-squared</li>
  *   <li>Adjusted R-squared</li>
  *   <li>F-Test</li>
  *   <li>Akaike Information Criterion (AIC)</li></ul></li>
  * <li>For each coefficient:<ul>
  *   <li>Standard Error (SE)</li>
  *   <li>T-statistic</li>
  *   <li>P-value</li>
  *   <li>Confidence Interval</li></ul></li></ul>
  *
  * @param X The independent variable training data in DATASET(NumericField) format.
  *          Each observation (e.g. record) is identified by
  *          'id', and each feature is identified by field number (i.e.
  *          'number').  Omit this parameter when predicting from a persisted
  *          model.
  *
  * @param Y The dependent variable training data in DATASET(NumericField)
  *         format. Each observation (e.g. record) is identified by
  *         'id', and each feature is identified by field number. Omit this parameter when
  *         predicting from a persisted model.
  * @return The instantiated model.
  * @see ML_Core.Types.NumericField
  */
EXPORT OLS(DATASET(NumericField) X=empty_data, DATASET(NumericField) Y=empty_data)
                     := MODULE(ML_Core.Interfaces.IRegression())
  // Convert X to matrix form
  mX0 := pbbConverted.NFToMatrix(X);
  // Insert column of ones for Y intercept
  mX := MatUtils.InsertCols(mX0, 1, 1.0);
  // Convert Y to matrix form
  mY := pbbConverted.NFToMatrix(Y);
  DATASET(Layout_Cell) learnBetas := FUNCTION
    // OLS formula is Beta = (XtX)^-1 * XtY
    XtX := PBblas.gemm(TRUE, FALSE, 1.0, mX, mX);
    XtY := PBblas.gemm(TRUE, FALSE, 1.0, mX, mY);
    // We want to solve for Beta so we multiply both sides by XtX
    // and get XtX * Beta = XtY.  This is in AX = B form, so we
    // can solve it.
    // We can factor XtX and use the triangle solver (3 steps) to find the Beta
    // solution.
    // 1) Find Lower triangular factor of XtX (i.e. L) using Cholesky factorization
    L   := PBblas.potrf(triangle.Lower, XtX);
    // 2) Since L * U * X = B, we solve for L * (UX) = B and calculate S = UX
    S  := PBblas.trsm(Side.Ax, triangle.Lower, FALSE, diagonal.NotUnitTri, 1.0,
                          L, XtY);
    // 3) Now we know that U * X = S, so we can solve for X.  Notice that L-transposed is
    //    used in place of U (transpose flag is TRUE).  That is equivalent, because for
    //    real numbers Cholesky factors U = L**T (L-transposed).
    B:= PBblas.trsm(Side.Ax, triangle.Upper, TRUE, diagonal.NotUnitTri, 1.0,
                          L, S);
    return B;
  END;
  mBetas := learnBetas;
  Layout_Model betasToModel(Layout_Cell b) := TRANSFORM
    SELF.wi := b.wi_id;
    SELF.id := b.y;
    SELF.number    := b.x;
    SELF.value     := b.v;
  END;
  /**
    * Return the learned model that maps Independent X variables to
    * Dependent variable(s) Y.
    * <p>In the case of
    * OLS, the model represents a set of Betas which are the coefficients
    * of the linear model: Beta0 * 1 + Beta1 * Field1 + Beta2 * Field2 ...
    * The ID of each model record specifies to which Y variable the coefficient
    * applies.
    * <p>The Field Number ('number') indicates to which field of X the beta is to be
    * applied.  Field number 1 provides the intercept portion of the linear
    * model and is always multiplied by 1.
    * <p>Note that if multiple work-items are provided within X and Y, there
    * will be multiple models returned in one dataset. The models can be separated by
    * their work item id (i.e. 'wi').  A single model can be extracted from
    * a myriad model by using e.g., model(wi=myWI_id).
    * <p>GetModel should not
    * be called when predicting using a previously persisted model (i.e. when
    * training data was not passed to the module).
    *
    * <p>Example:
    * <pre> myModel := OLS(X, Y).GetModel;</pre>
    * @return  Model in DATASET(Layout_Model) format.
    * @see ML_core/Types.Layout_Model
    */
  EXPORT DATASET(Layout_Model) GetModel  := PROJECT(mBetas, betasToModel(LEFT));

  // Get betas from the model
  // Convert model to a row vector of betas
  Layout_Cell modelToBetas(Layout_Model m) := TRANSFORM
    SELF.wi_id := m.wi;
    SELF.x     := m.number;
    SELF.y     := m.id;
    SELF.v     := m.value;
  END;

  SHARED DATASET(Layout_Cell) mBetas(DATASET(Layout_Model) model=GetModel) :=
                                                PROJECT(model, modelToBetas(LEFT));
  /**
    * <p>Extract Beta values from the model.  Can be used during training and prediction
    * phases.
    * <p>For use during training phase, the 'model' parameter can be omitted.
    * GetModel will be called to retrieve the model based on the training data. <p>For
    * use during prediction phase, a previously persisted model should be provided.
    * <p>The 'number' field of the returned NumericField records specifies to which Y
    * the coefficient applies.
    * <p>The 'id' field of the returned record indicates the position of the Beta
    * value. ID = 1 provides the Beta for the constant term (i.e. the Y intercept) while
    * subsequent values reflect the Beta for each correspondingly numbered X feature.
    * Feature 1 corresponds to Beta with 'id' = 2 and so on.
    * <p>If 'model' contains multiple work-items, Separate sets of Betas will be returned
    * for each of the 'myriad' models (distinguished by 'wi').
    *
    * @param model Optional parameter provides a model that was previously retrieved
    *               using GetModel.  If omitted, GetModel will be used as the model.
    * @return DATASET(NumericField) containing the Beta values.
    * @see ML_Core.Tyeps.NumericField
    */
  EXPORT DATASET(NumericField) Betas(DATASET(Layout_Model) model=GetModel) := FUNCTION
    nfBetas := PbbConverted.MatrixToNF(mBetas(model));
    return nfBetas;
  END;

  /**
    * Predict the dependent variable values (Y) for any set of independent
    * variables (X).  Returns a predicted Y values for each observation
    * (i.e. record) of X.
    *
    * <p>This attribute supports the 'myriad' style interface in that multiple independent
    * work items may be present in 'newX', and multiple independent models may
    * be provided in 'model'.  The resulting predicted values will also be
    * separable by work item (i.e. wi).
    *
    * @param newX   The set of observations of independent variables in
    *               DATASET(NumericField) format.
    * @param model  Optional.  A model that was previously returned from GetModel (above).
    *               Note that a model from a previous run will only be valid
    *               if the field numbers in X are the same as when the model
    *               was learned.  If this parameter is omitted, the current
    *               model will be used.
    * @return       An estimation of the corresponding Y value for each
    *               observation of newX.  Returned in DATASET(NumericField)
    *               format with field number (i.e. 'number') indicating the
    *               dependent variable that is predicted.
    */
  EXPORT DATASET(NumericField) Predict(DATASET(NumericField) newX,
            DATASET(Layout_Model) model=GetModel) := FUNCTION
    mNewX := pbbConverted.NFToMatrix(newX);
    // Insert a column of ones for the Y intercept term
    mExtX := MatUtils.InsertCols(mNewX, 1, 1.0);
    // Multiply the new X values by Beta
    mNewY := PBblas.gemm(FALSE, FALSE, 1.0, mExtX, mBetas(model));
    NewY := pbbConverted.MatrixToNF(mNewY);
    return NewY;
  END;

  // The predicted values of Y using the given X
  mod := GetModel;
  SHARED DATASET(NumericField) modelY := Predict(X, mod);

  // Calculate the correlation coefficients (pearson) between
  // y and modelY (aka Yhat)
  SHARED Yhat := modelY;
  // Make record with Y * Yhat (i.e. YYhat)
  NumericField make_YYhat(NumericField l, NumericField r) := TRANSFORM
    SELF.value := l.value * r.value;
    SELF       := l;  // Arbitrary
  END;
  SHARED YYhat := JOIN(Y, Yhat, LEFT.wi=RIGHT.wi AND RIGHT.id=LEFT.id AND LEFT.number=RIGHT.number,
                    make_YYhat(LEFT, RIGHT));
  // Calculate aggregates for Y, Yhat, and YYhat
  SHARED Y_stats := ML_Core.FieldAggregates(Y).simple;
  Yhat_stats := ML_Core.FieldAggregates(Yhat).simple;
  YYhat_stats := ML_Core.FieldAggregates(YYhat).simple;
  // Composite record of needed Y and Yhat stats
  CompRec0 := RECORD
    t_work_item   wi;
    t_FieldNumber number;
    t_FieldReal   e_Y;
    t_FieldReal   e_Yhat;
    t_FieldReal   sd_Y;
    t_FieldReal   sd_Yhat;
  END;
  //  Composite record of Y, Yhat, and YYhat stats
  CompRec := RECORD(CompRec0)
    t_FieldReal   e_YYhat;
  END;
  // Build the composite records
  CompRec0 make_comp0(Y_stats le, Yhat_stats ri) := TRANSFORM
    SELF.e_Y     := le.mean;
    SELF.e_Yhat  := ri.mean;
    SELF.sd_Y    := le.sd;
    SELF.sd_Yhat := ri.sd;
    SELF         := le;
  END;
  Ycomp0 := JOIN(Y_stats, Yhat_stats, LEFT.wi=RIGHT.wi AND LEFT.number=RIGHT.number, make_comp0(LEFT, RIGHT));

  CompRec make_comp(CompRec0 le, YYhat_stats ri) := TRANSFORM
    SELF.e_YYhat := ri.mean;
    SELF         := le;
  END;

  SHARED Ycomp  := JOIN(Ycomp0, YYhat_stats, LEFT.wi=RIGHT.wi and LEFT.number=RIGHT.number, make_comp(LEFT, RIGHT));

  // Correlation Coefficient Record
  SHARED CoCoRec := RECORD
    t_work_item   wi;
    t_FieldNumber number;
    Types.t_FieldReal   covariance;
    Types.t_FieldReal   pearson;
  END;

  // Form the Correlation Coefficient Records from the composite records
  CoCoRec MakeCoCo(Ycomp lr) := TRANSFORM
    SELF.covariance := (lr.e_YYhat - lr.e_Y*lr.e_Yhat);
    SELF.pearson := SELF.covariance/(lr.sd_Y*lr.sd_Yhat);
    SELF := lr;
  END;

  SHARED cor_coefs := PROJECT(Ycomp, MakeCoCo(LEFT));

  // The R Squared values for the parameters
  SHARED R2Rec := RECORD
    t_work_item wi;
    t_fieldNumber number;
    t_FieldReal   RSquared;
  END;

  SHARED R2Rec makeRSQ(CoCoRec coco) := TRANSFORM
    SELF.wi := coco.wi;
    SELF.number := coco.number;
    SELF.RSquared := coco.pearson * coco.pearson;
  END;
  /**
    * Calculate the R-Squared Metric used to assess the fit of the regression line to the
    * training data. <p>Since the regression has chosen the best (i.e. least squared error) line
    * matching the data, this can be thought of as a measurement of the linearity of the
    * training data.  <p>R Squared generally varies between 0 and 1, with 1 indicating an exact
    * linear fit, and 0 indicating that a linear fit will have no predictive power.  Negative
    * values are possible under certain conditions, and indicate that the mean(Y) will be more
    * predictive than any linear fit.  <p>Moderate values of R squared (e.g. .5) may indicate
    * that the relationship of X -> Y is non-linear, or that the measurement error is high
    * relative to the linear correlation (e.g. many outliers).  In the former case, increasing
    * the dimensionality of X, such as by using polynomial variants of the features, may
    * yield a better fit.
    *
    * <p>R squared always increases when additional independent variables are added, so it should
    * not be used to determine the optimal set of X variables to include.  For that purpose,
    * use Adjusted R Squared (below) which penalizes larger numbers of variables.
    *
    * <p>Note that the result of this call is only meaningful during training phase (use case 1
    * above) as it is an analysis based on the training data which is not provided during a
    * prediction-only phase.
    *
    * @return  DATASET(R2Rec) with one record per dependent variable, per work-item.  The
    *          number field indicates the dependent variable and coresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    *
    */
  EXPORT DATASET(R2Rec)  RSquared := PROJECT(cor_coefs, makeRSQ(LEFT));

  // Produce an Analysis of Variance report
  // Get the number of observations for each feature
  card := ML_Core.FieldAggregates(X).Cardinality;
  // Use that to calculate the number of features (k) for each work-item
  SHARED fieldCounts := TABLE(card, {wi, k := COUNT(GROUP)}, wi);

  // Get basic stats for Y for each work-item
  SHARED tmpRec := RECORD
    RECORDOF(Y_stats);
    Types.t_fieldreal  RSquared;
    UNSIGNED           K; // Number of Independent variables
  END;
  // Combine the basic Y stats with R Squared
  Y_stats1 := JOIN(Y_stats, RSquared, LEFT.wi=RIGHT.wi AND LEFT.number = RIGHT.number,
          TRANSFORM(tmpRec,  SELF.RSquared := RIGHT.RSquared, SELF.K := 0, SELF := LEFT));
  // Include the number of dependent variables for each work-item (i.e. K)
  SHARED Y_stats2 := JOIN(Y_stats1, fieldCounts, LEFT.wi=RIGHT.wi,
                            TRANSFORM(tmpRec, SELF.K := RIGHT.K, SELF := LEFT), LOOKUP);
  /**
    * Record layout for the information returned from calcAnova.
    *
    * @field wi The work-item number
    * @field number The dependent field (i.e. regressor) number.
    * @field Total_SS The total Sum-of-Squares variance.
    * @field Model_SS The Sum-of-Squares variance explained by the model.
    * @field Error_SS The Sum-of-Squares variance not explained by the model (
    *                 aka Residual Sum-of-Squares).
    * @field Total_DF Degrees of freedom in the data.
    * @field Model_DF Degrees of freedom of the model.
    * @field Error_DF Degrees of freedom of the error.
    * @field Total_MS Mean square variance of the data.
    * @field Model_MS Mean square variance of the model.
    * @field Error_MS Mean square residual error.
    * @field Model_F The F-test statistic.
    */
  EXPORT AnovaRec := RECORD
    t_work_item           wi;
    t_fieldNumber         number;   // Y field number
    Types.t_fieldreal     Total_SS; // Sum of Squares
    Types.t_fieldreal     Model_SS; // Sum of Squares
    Types.t_fieldreal     Error_SS; // Sum of Squares
    Types.t_RecordID      Total_DF; // Degrees of Freedom
    Types.t_RecordID      Model_DF; // Degrees of Freedom
    Types.t_RecordID      Error_DF; // Degrees of Freedom
    Types.t_fieldreal     Total_MS; // Mean Square
    Types.t_fieldreal     Model_MS; // Mean Square
    Types.t_fieldreal     Error_MS; // Mean Square
    Types.t_fieldreal     Model_F;  // F-test Statistic
  END;

  SHARED AnovaRec calcAnova(tmpRec le) :=TRANSFORM
    k   := le.K;
    // SST = sample-variance * Total_DF.  Var gives population-variance
    // To convert population to sample, we multiply var by n/(n-1).  Since Total_DF
    // is (n-1) we end up with SST = pop-variance * n / (n-1) * (n-1) which
    // simplifies to SST = pop-variance * n
    SST := le.var*le.countval;
    SSM := SST*le.RSquared;
    SELF.wi       := le.wi;
    SELF.number   := le.number;
    SELF.Total_SS := SST;
    SELF.Model_SS := SSM;
    SELF.Error_SS := SST - SSM;
    SELF.Total_DF := le.countval-1;
    SELF.Model_DF := k;
    SELF.Error_DF := le.countval-k-1;
    SELF.Total_MS  := SELF.Total_SS / SELF.Total_DF;
    SELF.Model_MS := SELF.Model_SS / SELF.Model_DF;
    SELF.Error_MS := SELF.Error_SS / SELF.Error_DF;
    SELF.Model_F  := SELF.Model_MS/SELF.Error_MS;
  END;

  /**
    * ANOVA (Analysis of Variance) report.
    *
    * <p>Analyze the sources of variance.
    * <p>Basic ANOVA equality:  Model + Error = Total
    * <p>Determines how much of the variance of Y is explained by the
    * regression model, versus how much is due to the error term
    * (i.e. unexplained variance).
    * <p>This attribute is only meaningful during the training phase.
    *
    * <p>Provides one record per  work-item.
    * Each record provides the following statistics:<ul>
    *   <li>Total_SS -- Total Sum of Squares (SS) variance of the dependent data.</li>
    *   <li>Model_SS -- The SS variance represented within the model.</li>
    *   <li>Error_SS -- The SS variance not reflected by the model
    *                (i.e. Total_SS - Error_SS).</li>
    *   <li>Total_DF -- The total degrees of freedom within the dependent data.</li>
    *   <li>Model_DF -- Degrees of freedom of the model.</li>
    *   <li>Error_DF -- Degrees of freedom of the error component.</li>
    *   <li>Total_MS -- The Mean Square (MS) variance of the dependent data.</li>
    *   <li>Model_MS -- The Mean Square (MS) variance represented within the
    *                model.</li>
    *   <li>Error_MS -- The MS variance not reflected by the model.</li>
    *   <li>Model_F  -- The F-Test statistic: Model_MS / Error_MS.</li></ul>
    *
    * @return  DATASET(AnovaRec), one per work-item per dependent (Y) variable
    *          The number field indicates the dependent variable to which the
    *          analysis applies.
    * @see AnovaRec
    *
    */
  EXPORT Anova := FUNCTION
    a := PROJECT(Y_stats2, calcAnova(LEFT));
    return a;
  END;

  // Compute the covariance matrix aka variance-covariance matrix of coefficients
  // Convert to matrix form for computation
  // The calculation is (X**TX)**-1 * ANOVA.Error_MS
  mXorig := PBblas.Converted.NFToMatrix(X); // Matrix of X values
  // We extend X by inserting a column of ones
  mX    := MatUtils.InsertCols(mXorig, 1, 1.0);
  Layout_Cell make_ident(NumericField b) := TRANSFORM
    SELF.wi_id := b.wi;
    SELF.x := b.id;
    SELF.y := b.id;
    SELF.v := 1;
  END;
  ident := PROJECT(Betas(), make_ident(LEFT));
  // Calculate X transposed times X.  If the original X had shape N x M,
  // our extended XTX will have shape M+1 x M+1, with one row and column per
  // coefficient.
  mXTX := PBblas.gemm(TRUE, FALSE, 1.0, mX, mX);
  // Solve for: XTX * X**-1 * () = I)
  // Factor and two triangle solves gives us the solution for XTX**-1
  mXTX_fact := PBblas.potrf(triangle.Lower, mXTX);
  mXTX_S := PBblas.trsm(Side.Ax, Triangle.Lower, FALSE,
                       diagonal.NotUnitTri, 1.0, mXTX_fact, ident);
  SHARED mXTXinv := PBblas.trsm(Side.Ax, triangle.Upper, TRUE,
                       diagonal.NotUnitTri, 1.0, mXTX_fact, mXTX_S);

  // Scale each cell by multiplying by the ANOVA Mean Square Error to arrive
  // at the covariance matrix of the coefficients.
  Layout_Cell scale_vc(Layout_Cell l, Anova r) := TRANSFORM
    SELF.v := l.v * r.Error_MS;
    SELF   := l;
  END;
  mCoef_covar(y_number) := JOIN(mXTXinv, Anova(number=y_number), LEFT.wi_id = RIGHT.wi, scale_vc(LEFT, RIGHT));

  /**
    * Coefficient Variance-Covariance Matrix
    *
    * This is the covariance matrix of the coefficients (i.e. Betas), not to be confused
    * with the covariance matrix for X.
    * If X is an N x M matrix (i.e. N observations of M features), then this will
    * be an M+1 x M+1 matrix, since there are M + 1 coefficients including the Y intercept.
    * Index 1 represents the Y intercept, Index 2 represents feature 1 .. Index M+1 represents
    * Feature M.  The diagonal entries designate the sampling variance of the coefficient,
    * while the non-diagonal entries represents the covariance between two
    * coefficients.  For example, entry 2,3 represents the covariance between the coefficients
    * for feature 1 and feature 2.
    *
    * This supports the myriad interface and will produce separate matrices for each
    * work-item specified within X and Y.
    *
    * Note that, since this returns a matrix of coefs x coefs, it can only provide the
    * matrix for one dependent variable at a time.  Therefore it needs to be called multiple
    * times in the case of multi-variate regression (more than one dependent variable).
    *
    * This is only meaningful during the training phase.
    *
    * @param y_number The number of the y variable whose covariance matrix to return
    * @return  DATASET(NumericField) representing the covariance matrix of the coefficients
    *
    */
  // Convert matrix form to numeric field form
  SHARED DATASET(NumericField) Coef_covar(t_FieldNumber y_number) := PBBConverted.MatrixToNF(mCoef_covar(y_number));


  DATASET(NumericField) make_coef_var := FUNCTION
    NumericField make_covar(NumericField b, AnovaRec a) := TRANSFORM
      SELF.id     := b.id;
      SELF.number := b.number;
      covar       := mXTXinv(wi_id=b.wi AND x=b.id AND y = b.id)[1].v * a.Error_MS;
      SELF.value  := covar;
      SELF        := b;
    END;
    covar := JOIN(Betas(), anova, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number, make_covar(LEFT, RIGHT), LOOKUP);
    return covar;
  END;
  // Make a matrix where rows are the coefficient number and columns are the dependent (Y) variable number
  // Values are the variance of the coefficient for the given Y.
  SHARED Coef_var := make_coef_var;

  SHARED NumericField calc_sErr(NumericField lr) := TRANSFORM
    SELF.value := SQRT(lr.value);
    SELF       := lr;
  END;

  /**
    * Compute the Standard Error of the Regression Coefficients.
    *
    * <p>Describes the variability of the regression error for each coefficient.
    *
    * <p>Only meaningful during the training phase.
    *
    * @return DATASET(NumericField), one record per Beta coefficient per dependent
    *         variable per work-item.
    *         The 'id' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc.
    *         The 'number' field indicates the dependent variable to which the
    *         coefficient applies.
    */
  EXPORT DATASET(NumericField) SE := FUNCTION
    // Standard error of the regression coefficients is just the square root of the
    // sampling variance of each coefficient (i.e. the diagonal terms of the Coefficient
    // Covariance Matrix).
    sErr := PROJECT(Coef_var, calc_sErr(LEFT));
    RETURN sErr;
  END;

  // T-Statistic
  // Transformation to calculate the T Statistic
  NumericField tStat_transform(NumericField b, NumericField s) := TRANSFORM
    SELF.value := b.value / s.value;
    SELF := b;
  END;
  /**
    * Compute the T-Statistic.
    *
    * <p>The T-statistic identifies the significance of the value of each regression
    * coefficient.  Its calculation is simply the value of the coefficient divided
    * by the Standard Error of the coefficient.  A larger absolute value of the
    * T-statistic indicates that the coefficient is more significant.
    *
    * <p>Only meaningful during the training phase.
    *
    * @return DATSET(NumericField), one record per Beta coefficient per dependent
    *         variable per work-item.
    *         The 'id' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc.
    *         The number field indicates the dependent variable to which the
    *         coefficient applies.
    */
  EXPORT DATASET(NumericField) TStat := JOIN(Betas(), SE, LEFT.wi = RIGHT.wi AND
                                    LEFT.id = RIGHT.id AND LEFT.number = right.number,
                                    tStat_transform(LEFT, RIGHT));

  // Calculate Adjusted R Squared
  R2Rec adjustR2(Rsquared l, Anova r) := TRANSFORM
    SELF.RSquared := 1 - ( 1 - l.RSquared) * (r.Total_DF / r.Error_DF);
    SELF          := l;
  END;

  /**
    * Calculate Adjusted R Squared, a normalized version of R Squared
    * that does not arbitrarily increase with the number of features.
    * <p>Adjusted R2, rather than R2 should always be used when trying to determine the
    * best set of features to include in a model.  When adding features, R2 will
    * always increase, whether or not it improves the predictive power of the model.
    * Adjusted R2, however, will only increase with the predictive power of the model.
    *
    * @return  DATASET(R2Rec), one record per dependent variable per work-item. The
    *          number field indicates the dependent variable and corresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    *
    */
  EXPORT DATASET(R2Rec) AdjRSquared := JOIN(Rsquared, Anova,
                          LEFT.wi=RIGHT.wi AND LEFT.number = RIGHT.number,
                          adjustR2(LEFT, RIGHT));

  SHARED PI := 3.141592653589793;

  // Record format for AIC results
  /**
    * Record layout for return from AIC function.
    *
    * @field wi The work-item number.
    * @field number The field-number of the dependent variable (i.e. regressor).
    * @field AIC The Akaike Information Criteria value.
    */
  EXPORT AICRec := RECORD
    t_work_item wi;
    t_FieldNumber number;
    Types.t_FieldReal AIC;
  END;

  AICRec calc_aic(AnovaRec a) := TRANSFORM
    n := a.Total_DF + 1; // Number of samples
    p := a.Model_DF + 1;  // Number of features
    // AIC Calculation derived as follows:
    // AIC = -2LL + 2p where LL is the Log Likelihood of Beta
    // LL = -N/2 * LN(2*PI) - N/2 * LN(sigma^2) - 1/(2*sigma^2) * SUM((y-yhat)^2)
    //    = -N/2 * LN(2*PI) - N/2 * LN(Error_SS/N) - N/(2*Error_SS) * Error_SS  // Note: sigma^2 = Error_SS/N, SUM((y-yhat)^2 = Error_SS
    //    = -N/2 * LN(2*PI) - N/2 * LN(Error_SS/N) - N/2
    // Therefore,
    // AIC = -2 (-N/2 * LN(2*PI) - N/2 * LN(Error_SS/N) - N/2) + 2p
    //     = N * LN(2*PI) + N * LN(Error_SS/N) + N
    //     = N * (LN(2*PI) + 1 + LN(Error_SS/N)) + 2*p
    SELF.AIC := n * (LN(2*PI) + 1 + LN(a.Error_SS / n)) + 2 * p;
    SELF.number := a.number;
    SELF.wi  := a.wi;
  END;

  /**
    * Calculate the Akaike Information Criterion (AIC).
    *
    * <p>AIC is an Information Theory based criterion for assessing Goodness of Fit (GoF).
    * <p>Lower values mean better fit.
    *
    * @return DATASET(AICRec), one record per dependent variable per work-item. The
    *          number field indicates the dependent variable and corresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    *
    */
  EXPORT DATASET(AICRec) AIC := PROJECT(Anova, calc_aic(LEFT));


  // ********* START Temporary Statistical Distribution Section
  // We temporarily include code here for needed statistical distributions.
  // This should be removed once we have a productized Distribution module.

  /** Density vector
    * @internal
    */
  EXPORT RangeVec := RECORD
    t_Count RangeNumber;
    t_FieldReal RangeLow; // Values > RangeLow
    t_FieldReal RangeHigh; // Values <= RangeHigh
    t_FieldReal P;
  END;
  SHARED E := 2.718281828459;
  /**
    * Base class for probability distribution attributes.
    * @internal
    */
  EXPORT DistributionBase(t_Count Nranges = 10000) := MODULE,VIRTUAL
    EXPORT Low := 0;  // Override for each distribution
    EXPORT High := 100; // OVerride for each distribution
    EXPORT t_FieldReal Density(t_FieldReal t) := 0.0; // Density function at stated point
    EXPORT RangeWidth := (High - Low) / Nranges;
    EXPORT DATASET(RangeVec) DensityV() := FUNCTION
      dummy := DATASET([{0,0,0,0}], RangeVec);
      RangeVec make_density(RangeVec d, UNSIGNED c) := TRANSFORM
        SELF.RangeNumber := c;
        SELF.RangeLow := Low + (c-1) * RangeWidth;
        SELF.RangeHigh := SELF.RangeLow + RangeWidth;
        SELF.P := Density((SELF.RangeLow + SELF.RangeHigh) / 2);
      END;
      vec := NORMALIZE(dummy, Nranges, make_density(LEFT, COUNTER));
      RETURN vec;
    END;
    // Default CumulativeV works by simple numerical integration of the DensityVec
    EXPORT CumulativeV() := FUNCTION
      d := DensityV();
      RangeVec Accum(RangeVec le,RangeVec ri) := TRANSFORM
        SELF.p := le.p+ri.p*RangeWidth;
        SELF := ri;
      END;
      RETURN ITERATE(d,Accum(LEFT,RIGHT));
    END;
    midRange(RangeVec rv) := (rv.RangeLow + rv.RangeHigh) / 2;
    // Interpolate within the range
    SHARED InterC(RangeVec rv, t_FieldReal t) := rv.P - Density(midRange(rv))*(rv.RangeHigh - t);;

    // Default Cumulative works from the Cumulative Vector
    EXPORT t_FieldReal Cumulative(t_FieldReal t) :=FUNCTION // Cumulative probability at stated point
      cv := CumulativeV();
      RETURN MAP( t >= High => 1.0,
                  t <= Low => 0.0,
                  InterC(cv(t > RangeLow, t <= RangeHigh)[1], t));
    END;
    // Default NTile works from the Cumulative Vector
    //
    EXPORT t_FieldReal NTile(t_FieldReal Pc) :=FUNCTION // Value of the Pc percentile
      cv := CumulativeV();
      targetcv := cv(P>=PC)[1];
      RETURN MAP( Pc >= MAX(cv,P) => MAX(cv,RangeHigh),
                  Pc <= 0.0 => MIN(cv,RangeLow),
                   targetcv.RangeHigh);
    END;
    EXPORT InvDensity(t_FieldReal delta) := 0.0; //Only sensible for monotonic distributions
    EXPORT Discrete := FALSE;
  END;

  /** Student T distribution
    * This distribution is entirely symmetric about the mean - so we will model the >= 0 portion
    * Warning - v=1 tops out around the 99.5th percentile, 2+ is fine.
    * @internal
    */

  EXPORT TDistribution(t_Discrete v_in,t_Count NRanges = 10000) := MODULE(DistributionBase(NRanges))
    SHARED v := v_in;
    SHARED Multiplier := (1/(SQRT(v)*math.Beta(.5, v/2)));
    // Compute the value of t for which a given density is obtained
    SHARED LowDensity := 0.00001; // Go down as far as a density of 1x10-5
    EXPORT InvDensity(t_FieldReal delta) := SQRT(v*(EXP(LN(delta/Multiplier)*-2.0/(v+1))-1));
    // We are defining a high value as the value at which the density is 'too low to care'
    //EXPORT High := InvDensity(LowDensity);
    EXPORT High := 10;
    EXPORT Low := 0;
    EXPORT RangeWidth := (high-low)/NRanges;
    // Generating functions are responsible for making these in ascending order
    EXPORT t_FieldReal Density(t_FieldReal t) := Multiplier * POWER(1 + t * t / v,-0.5*(v+1) );
    EXPORT CumulativeV() := FUNCTION
      d := DensityV();
      // The general integration really doesn't work for v=1 and v=2 - fortunately there are 'nice' closed forms for the CDF for those values of v
      RangeVec Accum(RangeVec le,RangeVec ri) := TRANSFORM
        SELF.p := MAP( v = 1 => 0.5+ATAN(ri.RangeHigh)/PI, // Special case CDF for v = 1
                       v = 2 => (1+ri.RangeHigh/SQRT(2+POWER(ri.RangeHigh,2)))/2, // Special case of CDF for v=2
                       IF(le.p=0,0.5,le.p)+ ri.p*RangeWidth );
        SELF := ri;
      END;

      RETURN ITERATE(d,Accum(LEFT,RIGHT)); // Global iterates are horrible - but this should be tiny
    END;
    // Cumulative works from the Cumulative Vector
    EXPORT t_FieldReal Cumulative(t_FieldReal t) :=FUNCTION // Cumulative probability at stated point
      cv := CumulativeV();
      // If the range high value is at an intermediate point of a range then interpolate the result\
      // Interpolation done as follows :
      // cumulative(RH) = cumulative(v.RangeHigh) - prob(RH <= x < v.Rangehigh)
      // prob(RH <= x < v.Rangehigh) =  Density((RH + Rangehigh)/2) * (RH - Rangehigh) [Rectangle Rule for integration]
      //InterC(RangeVec rv) := rv.P - Density((rv.RangeLow + rv.RangeHigh)/2)*(rv.RangeHigh - t);
      RETURN MAP( t >= High => 1.0,
                t <= Low => 0.0,
                InterC(cv(t > RangeLow,t <= RangeHigh)[1], t) );
    END;
  END;
  /**
    * F Distribution
    * @internal
    */
  EXPORT FDistribution(t_Discrete d1_in, t_Discrete d2_in, t_Count NRanges = 10000) := MODULE(DistributionBase(NRanges))
    SHARED REAL8 d1 := d1_in;
    SHARED REAL8 d2 := d2_in;
    EXPORT Low := 0;
    EXPORT High := 10;
    EXPORT RangeWidth := (high - low)/NRanges;
    // Method 1 -- Wikipedia
    SHARED REAL8 Multiplier :=  POWER(d1/d2, d1/2) / Math.Beta(d1/2, d2/2);
    EXPORT t_FieldReal Density(t_FieldReal t) := Multiplier * POWER(t, d1/2 - 1) *
                                               POWER(1 + d1 * t / d2, -(d1 + d2)/2);
    // Method 2 -- Univ Alabama -- http://www.math.uah.edu/stat/special/Fisher.html
    //SHARED Multiplier := d1 / (d2 * Math.Beta(d1/2, d2/2));
    //EXPORT t_FieldReal Density(t_FieldReal RH) := Multiplier * POWER(RH*d1/d2, d1/2-1) / POWER(1+RH*d1/d2,d1/2 + d2/2);
    // Method 3 -- Loyola University -- http://webpages.math.luc.edu/~jdg/w3teaching/stat_305/sp06/pdf/densityF.pdf
    //EXPORT t_FieldReal Density(t_FieldReal RH) :=
         //(Math.Gamma((d1 + d2)/2) * POWER(d1, d1/2) * POWER(d2, d2/2) * POWER(RH, d1/2 - 1)) /
         //(Math.Gamma(d1/2) * Math.Gamma(d2/2) * POWER(d1*RH + d2, (d1+d2)/2));
    // Method 4 -- dainielsoper.com
         //(SQRT(POWER(d1*RH, d1) * POWER(d2, d2)/(POWER(d1*RH + d2, d1+d2)))) /
         //(RH * Math.Beta(d1/2, d2/2));
  END;
  /** Normal Distribution.
    * @internal
    */
  EXPORT NormalDistribution(t_Count NRanges) := MODULE(DistributionBase(NRanges))
   EXPORT t_FieldReal Density(t_FieldReal t) := POWER(E, -.5 * t * t) / SQRT(2*PI);
  END;
  // ******** END Temporary Statistical Distribution Section

  // Calculate P-val
  NumericField calc_P(NumericField t, AnovaRec a) := TRANSFORM
    // Calculate the probability (using the T-distribution) of getting a T-statistic
    // less than T.
    tdist := TDistribution(a.Error_DF);
    probT := tdist.cumulative(ABS(t.value));

    //probT := 1; //stdtr(a.Error_DF, ABS(t.value));
    // The complement is the probability of a T-statistic > t.
    // Multiply the complement by 2 since this is a 2-tailed test.
    SELF.value := 2 * (1 - probT);
    SELF       := t;
  END;
  /**
    * Calculate the P-value for each coefficient, which is the probability that
    * the coefficient is insignificant (i.e. actually zero).  <p>A low P-value (e.g. .05)
    * provides evidence that the coefficient is significant in the model.  <p>A
    * high P-value indicates that the coefficient value should, in fact, be zero.
    * <p>P-value is related to the T-Statistic, and can be thought of as a normalized
    * version of the T-Statistic.
    *
    * <p>Only meaningful during the training phase.
    *
    * @return DATSET(NumericField), one record per Beta coefficient per dependent
    *         variable per work-item.
    *         The 'id' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc. The
    *          number field indicates the dependent variable and corresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    */
  EXPORT pVal := JOIN(tStat, Anova, LEFT.wi = RIGHT.wi, calc_P(LEFT, RIGHT));

  // Confidence Interval
  /**
    * Record layout for the return from ConfInt.
    *
    * @field wi The work-item number.
    * @field id The coefficient number (1 is the constant term).
    * @field number The dependent field number (i.e. regressor).
    * @field LowerInt The lower bound of the interval.
    * @field UpperInt The upper bound of the interval.
    */
  EXPORT ConfintRec := RECORD
    t_work_item    wi;
    t_RecordID     id;
    t_FieldNumber  number;
    t_Fieldreal    LowerInt;
    t_Fieldreal    UpperInt;
  END;

  // Transform to compute margins for each field
  NumericField calc_margins(NumericField s, AnovaRec a, REAL p) := TRANSFORM
    k      := a.Error_DF;  // Error degrees of freedom
    // Calculate the inverse cdf of the T-Distribution and multiply by the standard
    // error to obtain the margin about each beta
    td := TDistribution(k, 100000);
    margin := td.Ntile(p) * s.value;
    SELF.value := margin;
    SELF       := s;
  END;

  // Transform to compute the intervals from the Betas and Margins
  ConfintRec confint_transform(NumericField b, NumericField m) := TRANSFORM
    margin        := m.value;
    SELF.UpperInt := b.value + margin;
    SELF.LowerInt := b.value - margin;
    SELF := b;
  END;

  /**
    * Compute the Confidence Interval for each coefficient.
    *
    * <p>The Confidence Interval determines the upper and lower bounds of each estimated coefficient
    * given a confidence level (level) that is required.
    * <p>For example, one could say that there is a 95% probability (level) that the coefficient of the
    * first independent variable is between 2.05 and 3.62.  This allows error margins to be
    * determined with the desired confidence level.  If the confidence interval spans zero,
    * it implies that the coefficient may not be significant at the specified confidence level.
    *
    * @param level The level of confidence required, expressed as a percentage from 0.0 to 100.0.
    * @return DATASET(ConfintRec) with one record per coefficient per dependent variable
    *         per work-item.
    *         The 'id' field is the coefficient number, with 1 being the
    *         Y intercept, 2 being the coefficient for the first feature, etc.
    *         The number field indicates the dependent variable and corresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    */
  EXPORT ConfInt(Types.t_fieldReal level) := FUNCTION
    // Compute the probability level we are looking for
    // e.g. if 95% confidence level, each tail will contain 2.5%.  Therefore
    // we want to look for the p-value of .975
    p_level := 1 - ((1 - level / 100) / 2);
    // Compute the margins based on the T-Distribution and the Standard Errors of each Beta
    margins := JOIN(SE, Anova, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                     calc_margins(LEFT, RIGHT, p_level));
    // Use the margins computed above and the Betas themselves to compute the intervals
    RETURN JOIN(Betas(), margins,
                LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id AND LEFT.id = RIGHT.id
                     AND LEFT.number = RIGHT.number,
                     confint_transform(LEFT,RIGHT));
  END;

  // F-Test

  // Returned record for F-Test
  /**
    * The record layout for the return from Ftest
    * @field wi The work-item number.
    * @field number The dependent field (i.e. regressor) number.
    * @field Model_F The F-value.
    * @field pValue The p-value for the full distribution.
    */
  EXPORT FTestRec := RECORD
    t_work_item wi;
    t_FieldNumber number;
    t_FieldReal Model_F;
    t_FIeldReal pValue;
  END;

  // Caclulate F Probability (F-Test)
  FtestRec calc_F(AnovaRec a) := TRANSFORM
    // probF is the probability of seeing a value less than Model_F
    // fdtr calculates the cumulative distribution (i.e. cdf) of the F-distribution
    // with the degrees of freedom from the ANOVA model.
    fd := FDistribution(a.Model_DF, a.Error_DF);
    probF := fd.cumulative(a.Model_F);  //fdtr(a.Model_DF, a.Error_DF, a.Model_F);
    // The complement is the probability of seeing a value of Model_F or greater by
    // random chance, given the null hypothesis that alll of the coefficients are
    // actually zero.
    // Note: This is a one-tailed test
    SELF.pValue := 1 - probF;
    SELF.Model_F := a.Model_F;
    SELF.wi      := a.wi;
    SELF.number  := a.number;
  END;

  /**
    * Perform an F-test. <p>Calculate the P-value for the full regression, which is the probability that
    * all of the coefficients are insignificant (i.e. actually zero).
    * <p>A low P-value (e.g. .05)
    * provides evidence that at least one coefficient is significant.  A
    * high P-value indicates that all the coefficient values should in fact be zero,
    * implying that the regression has no statistically significant predictive power.
    * <p>P-value is related to the ANOVA F-Statistic, and can be thought of as a standardized
    * version of the ANOVA F-Statistic.
    *
    * <p>The F-Test and T-Test are similar, except that the T-test is used to test the
    * significance of each coefficient, while the F-Test is used to test the significance
    * of the entire regression.  For simple linear regression (i.e. only one independent
    * variable, the T-Test and F-Test are equivalent.
    *
    * @return DATASET(FTestRec), one record per dependent variable per work-item.
    *         The number field indicates the dependent variable and corresponds to the number
    *          field of the dependent (Y) variable to which it applies.
    */
  EXPORT DATASET(FTestRec) FTest := PROJECT(Anova, calc_F(LEFT));
END;
