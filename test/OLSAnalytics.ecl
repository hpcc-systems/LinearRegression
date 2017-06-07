/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
// This module performas a multi-variate regression and verifies that if the training data is the
// same for two dependents, the analytics for the two dependent variables will be the same, and that if the
// traning data is different, then the analytics are different.
// This is sufficient because OLSValidate verifies that in a Univariate case, the analytics are correct.
// So, if the second variable is the same as the first (with the same training data), then it is also
// correct.  The test of difference for the third dependent verifies that the data is not simply
// a duplicate.
// It also tests the temporary statistical distributions within the OLS module.  When those functions
// are deprecated, the corresponding tests in this module should also be removed.

IMPORT $.^ as LROLS;
IMPORT ML_Core;
IMPORT ML_Core.Types as mlTypes;
IMPORT PBblas;
IMPORT PBBlas.test.MakeTestMatrix as tm;
IMPORT PBBlas.Types as pbbTypes;
IMPORT PBBlas.Converted as pbbConverted;
IMPORT ML_Core.Math as Math;
Layout_Cell := pbbTypes.Layout_Cell;
NumericField := mlTypes.NumericField;
two31 := POWER(2, 31);  // 2**31

epsilon := .000001;  // Allowable error

REAL Noise(maxv=.1) := FUNCTION
  out := ((RANDOM()-two31)%1000000)/(1000000/maxv);
  return out;
END;

// Create a multiple linear regression with 3 independents, and also multi-variate
// with three dependents.
compX := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
END;
compX makeComposite(Layout_Cell l, DATASET(Layout_Cell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
END;
// Generation coefficients for Y1 and Y2
A1_1 := -1.8;
A2_1 := -.333;
A3_1 := 1.13;
B_1 := -3.333;
// Generation coefficients for Y3
A1_2 := .0013;
A2_2 := -.123;
A3_2 := .00015;
B_2 := -5.01;

N := 50;
M := 3;

mX := tm.RandomMatrix(N, M, 1.0, 2);

X := pbbConverted.MatrixToNF(mX);
gX := GROUP(mX, x, ALL);
cX := ROLLUP(gX,  GROUP, makeComposite(LEFT, ROWS(LEFT)));
Layout_Cell makeY(compX X, UNSIGNED c) := TRANSFORM
  SELF.x := X.id;
  SELF.y := IF(c=1, 1, 3);  // Make Y1 and Y3.  Y2 will be a copy of Y1,
  SELF.wi_id := X.wi;
  v1 := A1_1 * X.X1 + A2_1 * X.X2 + A3_1 * X.X3 + B_1 + Noise(1);
  v2 := A1_2 * X.X1 + A2_2 * X.X2 + A3_2 * X.X3 + B_2 + Noise(10000);
  SELF.v := IF(c=1, v1, v2);
END;
// Make 2 Y variables.  One with more noise than the other and different coefficients
mY := NORMALIZE(cX, 2, makeY(LEFT, COUNTER));
// Now copy Y1 to Y2
Layout_Cell make_Y2(Layout_Cell y_cell) := TRANSFORM
  SELF.y := IF(y_cell.y = 1, 2, SKIP);
  SELF   := y_cell
END;
mY2 := PROJECT(mY, make_Y2(LEFT));
mY3 := mY + mY2;
Y := pbbConverted.MatrixToNF(mY3);

// Create a regression with 3 independents and 3 dependents
lr := LROLS.OLS(X, Y);

// Test analytic functions starting with the most independent attributes, and working up
// from there

// Test output format
TestRec := RECORD
  STRING32 testname;
  UNSIGNED errors;
  STRING   details;
END;

INTEGER testSame(REAL8 v, REAL8 expected) := FUNCTION
 err := IF(ABS((v - expected)/expected) < epsilon, 0, 1);
 return err;
END;
INTEGER testDiff(REAL8 v, REAL8 expected) := FUNCTION
 err := IF(ABS((v - expected)/expected) >= epsilon, 0, 1);
 return err;
END;




// TEST01 -- R Squared
Rsq := lr.RSquared;
Rsq1 := Rsq(number=1)[1].RSquared;
Rsq2 := Rsq(number=2)[1].RSquared;
Rsq3 := Rsq(number=3)[1].RSquared;
errors1_1 := testSame(Rsq1, Rsq2);
errors1_2 := testDiff(Rsq1, Rsq3);
details1_1 := 'Rsq1: ' + Rsq1 + ', Rsq2: ' + Rsq2;
details1_2 := 'Rsq1: ' + Rsq1 + ', Rsq3: ' + Rsq3;
test1_1 := DATASET([{'TEST1_1 -- R Squared(1 vs 2)', errors1_1, details1_1}], TestRec);
test1_2 := DATASET([{'TEST1_2 -- R Squared(1 vs 3)', errors1_2, details1_2}], TestRec);

// TEST02 -- ANOVA
an := lr.Anova;
an1 := an(number=1)[1];
an2 := an(number=2)[1];
an3 := an(number=3)[1];
totalSS1 := an1.Total_SS;
totalSS2 := an2.Total_SS;
totalSS3 := an3.Total_SS;
modelSS1 := an1.Model_SS;
modelSS2 := an2.Model_SS;
modelSS3 := an3.Model_SS;
errorSS1 := an1.Error_SS;
errorSS2 := an2.Error_SS;
errorSS3 := an3.Error_SS;
totalDF1 := an1.Total_DF;
totalDF2 := an2.Total_DF;
totalDF3 := an3.Total_DF;
modelDF1 := an1.Model_DF;
modelDF2 := an2.Model_DF;
modelDF3 := an3.Model_DF;
errorDF1 := an1.Error_DF;
errorDF2 := an2.Error_DF;
errorDF3 := an3.Error_DF;
totalMS1 := an1.Total_MS;
totalMS2 := an2.Total_MS;
totalMS3 := an3.Total_MS;
modelMS1 := an1.Model_MS;
modelMS2 := an2.Model_MS;
modelMS3 := an3.Model_MS;
errorMS1 := an1.Error_MS;
errorMS2 := an2.Error_MS;
errorMS3 := an3.Error_MS;
modelF1  := an1.Model_F;
modelF2  := an2.Model_F;
modelF3 := an3.Model_F;
// 2_1 Total_SS
errors2_1_1 := testSame(TotalSS1, TotalSS2);
errors2_1_2 := testDiff(TotalSS1, TotalSS3);
details2_1_1 := 'totalSS1: ' + totalSS1 + ', totalSS2: ' + totalSS2;
details2_1_2 := 'totalSS1: ' + totalSS2 + ', totalSS3: ' + totalSS3;
test2_1_1 := DATASET([{'TEST2_1_1 -- Total_SS (1 vs 2)', errors2_1_1, details2_1_1}], TestRec);
test2_1_2 := DATASET([{'TEST2_1_2 -- Total_SS (1 vs 3)', errors2_1_2, details2_1_2}], TestRec);
// 2_2 Model_SS
errors2_2_1 := testSame(modelSS1, ModelSS2);
errors2_2_2 := testDiff(modelSS1, modelSS3);
details2_2_1 := 'modelSS1: ' + modelSS1 + ', modelSS2: ' + modelSS2;
details2_2_2 := 'modelSS1: ' + modelSS2 + ', modelSS3: ' + modelSS3;
test2_2_1 := DATASET([{'TEST2_2_1 -- Model_SS (1 vs 2)', errors2_2_1, details2_2_1}], TestRec);
test2_2_2 := DATASET([{'TEST2_2_2 -- Model_SS (1 vs 3)', errors2_2_2, details2_2_2}], TestRec);
// 2_3 Error_SS
errors2_3_1 := testSame(errorSS1, errorSS2);
errors2_3_2 := testDiff(errorSS1, errorSS3);
details2_3_1 := 'errorSS1: ' + errorSS1 + ', errorSS2: ' + errorSS2;
details2_3_2 := 'errorSS1: ' + errorSS2 + ', errrorSS3: ' + errorSS3;
test2_3_1 := DATASET([{'TEST2_3_1 -- Error_SS (1 vs 2)', errors2_3_1, details2_3_1}], TestRec);
test2_3_2 := DATASET([{'TEST2_3_2 -- Error_SS (1 vs 3)', errors2_3_2, details2_3_2}], TestRec);
// 2_4 Total_DF
// Note: for DF, only verify that all are the same
errors2_4_1 := testSame(TotalDF1, TotalDF2);
errors2_4_2 := testSame(TotalDF1, TotalDF3);
details2_4_1 := 'totalDF1: ' + totalDF1 + ', totalDF2: ' + totalDF2;
details2_4_2 := 'totalDF1: ' + totalDF2 + ', totalDF3: ' + totalDF3;
test2_4_1 := DATASET([{'TEST2_4_1 -- Total_DF (1 vs 2)', errors2_4_1, details2_4_1}], TestRec);
test2_4_2 := DATASET([{'TEST2_4_2 -- Total_DF (1 vs 3)', errors2_4_2, details2_4_2}], TestRec);
// 2_5 Model_DF
errors2_5_1 := testSame(modelDF1, ModelDF2);
errors2_5_2 := testSame(modelDF1, modelDF3);
details2_5_1 := 'modelDF1: ' + modelDF1 + ', modelDF2: ' + modelDF2;
details2_5_2 := 'modelDF1: ' + modelDF2 + ', modelDF3: ' + modelDF3;
test2_5_1 := DATASET([{'TEST2_5_1 -- Model_DF (1 vs 2)', errors2_5_1, details2_5_1}], TestRec);
test2_5_2 := DATASET([{'TEST2_5_2 -- Model_DF (1 vs 3)', errors2_5_2, details2_5_2}], TestRec);
// 2_6 Error_DF
errors2_6_1 := testSame(errorDF1, errorDF2);
errors2_6_2 := testSame(errorDF1, errorDF3);
details2_6_1 := 'errorDF1: ' + errorDF1 + ', errorDF2: ' + errorDF2;
details2_6_2 := 'errorDF1: ' + errorDF2 + ', errrorDF3: ' + errorDF3;
test2_6_1 := DATASET([{'TEST2_6_1 -- Error_DF (1 vs 2)', errors2_6_1, details2_6_1}], TestRec);
test2_6_2 := DATASET([{'TEST2_6_2 -- Error_DF (1 vs 3)', errors2_6_2, details2_6_2}], TestRec);
// 2_7 Total_MS
errors2_7_1 := testSame(TotalMS1, TotalMS2);
errors2_7_2 := testDiff(TotalMS1, TotalMS3);
details2_7_1 := 'totalMS1: ' + totalMS1 + ', totalMS2: ' + totalMS2;
details2_7_2 := 'totalMS1: ' + totalMS2 + ', totalMS3: ' + totalMS3;
test2_7_1 := DATASET([{'TEST2_7_1 -- Total_MS (1 vs 2)', errors2_7_1, details2_7_1}], TestRec);
test2_7_2 := DATASET([{'TEST2_7_2 -- Total_MS (1 vs 3)', errors2_7_2, details2_7_2}], TestRec);
// 2_8 Model_MS
errors2_8_1 := testSame(modelMS1, ModelMS2);
errors2_8_2 := testDiff(modelMS1, modelMS3);
details2_8_1 := 'modelMS1: ' + modelMS1 + ', modelMS2: ' + modelMS2;
details2_8_2 := 'modelMS1: ' + modelMS2 + ', modelMS3: ' + modelMS3;
test2_8_1 := DATASET([{'TEST2_8_1 -- Model_MS (1 vs 2)', errors2_8_1, details2_8_1}], TestRec);
test2_8_2 := DATASET([{'TEST2_8_2 -- Model_MS (1 vs 3)', errors2_8_2, details2_8_2}], TestRec);
// 2_9 Error_MS
errors2_9_1 := testSame(errorMS1, errorMS2);
errors2_9_2 := testDiff(errorMS1, errorMS3);
details2_9_1 := 'errorMS1: ' + errorMS1 + ', errorMS2: ' + errorMS2;
details2_9_2 := 'errorSS1: ' + errorMS2 + ', errrorMS3: ' + errorMS3;
test2_9_1 := DATASET([{'TEST2_9_1 -- Error_MS (1 vs 2)', errors2_9_1, details2_9_1}], TestRec);
test2_9_2 := DATASET([{'TEST2_9_2 -- Error_MS (1 vs 3)', errors2_9_2, details2_9_2}], TestRec);
// 2_10 Model_F
errors2_10_1 := testSame(modelF1, modelF2);
errors2_10_2 := testDiff(modelF1, modelF3);
details2_10_1 := 'modelF1: ' + modelF1 + ', modelF2: ' + modelF2;
details2_10_2 := 'modelF1: ' + modelF1 + ', modelF3: ' + modelF3;
test2_10_1 := DATASET([{'TEST2_10_1 -- Model_F (1 vs 2)', errors2_10_1, details2_10_1}], TestRec);
test2_10_2 := DATASET([{'TEST2_10_2 -- Model_F (1 vs 3)', errors2_10_2, details2_10_2}], TestRec);


// TEST03 -- Adjusted R Squared
aRsq := lr.AdjRSquared;
aRsq1 := aRsq(number=1)[1].RSquared;
aRsq2 := aRsq(number=2)[1].RSquared;
aRsq3 := aRsq(number=3)[1].RSquared;
details3_1 := 'adjRsq1: ' + aRsq1 + ', adjRsq2: ' + aRsq2;
details3_2 := 'adjRsq1: ' + aRsq1 + ', adjRsq3: ' + aRsq3;
errors3_1 := testSame(aRsq1, aRsq2);
errors3_2 := testDiff(aRsq1, aRsq3);
test3_1 := DATASET([{'TEST3_1 -- Adjusted R Squared(1 vs 2)', errors3_1, details3_1}], TestRec);
test3_2 := DATASET([{'TEST3_2 -- AdjustedR Squared(1 vs 3)', errors3_2, details3_2}], TestRec);

// TEST04 -- Standard Error
// Just test for coefficient 3.  If it works for that, it will work for the rest
se := lr.SE;
se1 := se(number=1 AND id=4)[1].value;
se2 := se(number=2 AND id=4)[1].value;
se3 := se(number=3 AND id=4)[1].value;
details4_1 := 'SE1: ' + se1 + ', SE2: ' + se2;
details4_2 := 'SE1: ' + se1 + ', SE3: ' + se3;
errors4_1 := testSame(se1, se2);
errors4_2 := testDiff(se1, se3);
test4_1 := DATASET([{'TEST4_1 -- Standard Error(1 vs 2)', errors4_1, details4_1}], TestRec);
test4_2 := DATASET([{'TEST4_2 -- Standard Error(1 vs 3)', errors4_2, details4_2}], TestRec);

// TEST05 -- Tstat
// Just test for coefficient 3.  If it works for that, it will work for the rest
ts := lr.Tstat;
ts1 := ts(number=1 AND id=4)[1].value;
ts2 := ts(number=2 AND id=4)[1].value;
ts3 := ts(number=3 AND id=4)[1].value;
details5_1 := 'Tstat1: ' + ts1 + ', Tstat2: ' + ts2;
details5_2 := 'Tstat1: ' + ts1 + ', Tstat3: ' + ts3;
errors5_1 := testSame(ts1, ts2);
errors5_2 := testDiff(ts1, ts3);
test5_1 := DATASET([{'TEST5_1 -- Tstat(1 vs 2)', errors5_1, details5_1}], TestRec);
test5_2 := DATASET([{'TEST5_2 -- Tstat(1 vs 3)', errors5_2, details5_2}], TestRec);

// TEST06 -- P-val
// Just test for coefficient 3.  If it works for that, it will work for the rest
pv := lr.Pval;
pv1 := pv(number=1 AND id=4)[1].value;
pv2 := pv(number=2 AND id=4)[1].value;
pv3 := pv(number=3 AND id=4)[1].value;
details6_1 := 'Pval1: ' + pv1 + ', Pval2: ' + pv2;
details6_2 := 'Pval1: ' + pv1 + ', Pval3: ' + pv3;
errors6_1 := testSame(pv1, pv2);
errors6_2 := testDiff(pv1, pv3);
test6_1 := DATASET([{'TEST6_1 -- Pval (1 vs 2)', errors6_1, details6_1}], TestRec);
test6_2 := DATASET([{'TEST6_2 -- Pval (1 vs 3)', errors6_2, details6_2}], TestRec);

// TEST07 -- AIC
aic := lr.AIC;
aic1 := aic(number=1)[1].aic;
aic2 := aic(number=2)[1].aic;
aic3 := aic(number=3)[1].aic;
errors7_1 := testSame(aic1, aic2);
errors7_2 := testDiff(aic1, aic3);
details7_1 := 'AIC1: ' + aic1 + 'AIC2: ' + aic2;
details7_2 := 'AIC1: ' + aic1 + 'AIC3: ' + aic3;
test7_1 := DATASET([{'TEST7_1 -- AIC (1 vs 2)', errors7_1, details7_1}], TestRec);
test7_2 := DATASET([{'TEST7_2 -- AIC (1 vs 3)', errors7_2, details7_2}], TestRec);

// TEST08 -- Confidence Interval
// Just test for coefficient 3.  If it works for that, it will work for the rest
ci := lr.Confint(95);
li1 := ci(number=1 AND id=4)[1].LowerInt;
li2 := ci(number=2 AND id=4)[1].LowerInt;
li3 := ci(number=3 AND id=4)[1].LowerInt;
ui1 := ci(number=1 AND id=4)[1].UpperInt;
ui2 := ci(number=2 AND id=4)[1].UpperInt;
ui3 := ci(number=3 AND id=4)[1].UpperInt;
details8_1_1 := 'LowerInt1: ' + li1 + ', LowerInt2: ' + li2;
details8_1_2 := 'LowerInt1: ' + li1 + ', LowerInt3: ' + li3;
details8_2_1 := 'UpperInt1: ' + ui1 + ', UpperInt2: ' + ui2;
details8_2_2 := 'UpperInt1: ' + ui1 + ', UpperInt3: ' + ui3;
errors8_1_1 := testSame(li1, li2);
errors8_1_2 := testDiff(li1, li3);
errors8_2_1 := testSame(ui1, ui2);
errors8_2_2 := testDiff(ui1, ui3);
test8_1_1 := DATASET([{'TEST8_1_1 -- Conf Int Lower (1 vs 2)', errors8_1_1, details8_1_1}], TestRec);
test8_1_2 := DATASET([{'TEST8_1_2 -- Conf Int Lower (1 vs 3)', errors8_1_2, details8_1_2}], TestRec);
test8_2_1 := DATASET([{'TEST8_2_1 -- Conf Int Upper (1 vs 2)', errors8_2_1, details8_2_1}], TestRec);
test8_2_2 := DATASET([{'TEST8_2_2 -- Conf Int Upper (1 vs 3)', errors8_2_2, details8_2_2}], TestRec);

// TEST09 -- FTest
ft := lr.Ftest;
ft1 := ft(number=1)[1].pValue;
ft2 := ft(number=2)[1].pValue;
ft3 := ft(number=3)[1].pValue;
errors9_1 := testSame(ft1, ft2);
errors9_2 := testDiff(ft1, ft3);
details9_1 := 'Ftest1: ' + ft1 + ', Ftest2: ' + ft2;
details9_2 := 'Ftest1: ' + ft1 + ', Ftest3: ' + ft3;
test9_1 := DATASET([{'TEST9_1 -- Ftest Pvalue (1 vs 2)', errors9_1, details9_1}], TestRec);
test9_2 := DATASET([{'TEST9_2 -- Ftest Pvalue (1 vs 3)', errors9_2, details9_2}], TestRec);

// TEST10 -- T Distribution

tdist1 := lr.TDistribution(10);
pdfT1 := tdist1.density(2.0);
cumT1 := tdist1.cumulative(2.0);
tdist2 := lr.TDistribution(4168);
pdfT2   := tdist2.density(2.0);
cumT2 := tdist2.cumulative(2.0);
expPdfT1 := 0.061145766321218202;
expPdfT2 := 0.054013619512625885;
expCumT1 := 0.96330598261462974;
expCumT2 := 0.97721748014745291;
tval1 := tdist1.Ntile(expCumT1);
tval2 := tdist2.Ntile(expCumT2);
expTval1 := 2.0;
expTval2 := 2.0;
errors10_1 := testSame(pdfT1, expPdfT1);
errors10_2 := testSame(pdfT2, expPdfT2);
errors10_3 := testSame(cumT1, expCumT1);
errors10_4 := testSame(cumT2, expCumT2);
errors10_5 := testSame(tval1, expTval1);
errors10_6 := testSame(tval2, expTval2);
details10_1 := 'T PDF(2)[10]: ' + pdfT1 + ', Expected: ' + expPdfT1;
details10_2 := 'T PDF(2)[4168]: ' + pdfT2 + ', Expected: ' + expPdfT2;
details10_3 := 'Cumulative T: ' + cumT1 + ', Expected: ' + expCumT1;
details10_4 := 'Cumulative T: ' + cumT2 + ', Expected: ' + expCumT2;
details10_5 := 'T value: ' + tval1 + ', Expected: ' + expTval1;
details10_6 := 'T value: ' + tval2 + ', Expected: ' + expTval2;
test10_1 := DATASET([{'TEST10_1 -- T-Distr PDF(1)(2.0)', errors10_1, details10_1}], TestRec);
test10_2 := DATASET([{'TEST10_2 -- T-Distr PDF(2)(2.0)', errors10_2, details10_2}], TestRec);
test10_3 := DATASET([{'TEST10_3 -- T-Distr CDF(1)', errors10_3, details10_3}], TestRec);
test10_4 := DATASET([{'TEST10_4 -- T-Distr CDF(2)', errors10_4, details10_4}], TestRec);
test10_5 := DATASET([{'TEST10_5 -- T-Distr Inv CDF(1)', errors10_5, details10_5}], TestRec);
test10_6 := DATASET([{'TEST10_6 -- T-Distr Inv CDF(2)', errors10_6, details10_6}], TestRec);

// TEST11 -- F Distribution
fdist1 := lr.FDistribution(5, 10);
fdist2 := lr.FDistribution(8, 4168);
pdfF1 := fdist1.Density(2.0);
pdfF2 := fdist2.Density(2.0);
cumF1 := fdist1.cumulative(2.0);
cumF2 := fdist2.cumulative(2.0);
expPdfF1 := 0.16200574218011515;
expPdfF2 := 0.11483341094261676;
expCumF1 := 0.83580504910026132;
expCumF2 := 0.95734506461383584;
errors11_1 := testSame(pdfF1, expPdfF1);
errors11_2 := testSame(pdfF2, expPdfF2);
errors11_3 := testSame(cumF1, expCumF1);
errors11_4 := testSame(cumF2, expCumF2);
details11_1 := 'PDF(2.0): ' + pdfF1 + ', Expected: ' + expPdfF1;
details11_2 := 'PDF(2.0): ' + pdfF2 + ', Expected: ' + expPdfF2;
details11_3 := 'Cumulative F: ' + cumF1 + ', Expected: ' + expCumF1;
details11_4 := 'Cumulative F: ' + cumF2 + ', Expected: ' + expCumF2;
test11_1 := DATASET([{'TEST11_1 -- F-Distr PDF(1)', errors11_1, details11_1}], TestRec);
test11_2 := DATASET([{'TEST11_2 -- F-Distr PDF(2)', errors11_2, details11_2}], TestRec);
test11_3 := DATASET([{'TEST11_3 -- F-Distr CDF(1)', errors11_3, details11_3}], TestRec);
test11_4 := DATASET([{'TEST11_4 -- F-Distr CDF(2)', errors11_4, details11_4}], TestRec);
test11_5 := DATASET([{'TEST11_5 -- Beta(4, 2084)', 0, 'Beta val: ' + math.Beta(4, 2084)}], TestRec);

summary :=    test1_1 + test1_2
            + test2_1_1 + test2_1_2 + test2_2_1 + test2_2_2 + test2_3_1 + test2_3_2 + test2_4_1 + test2_4_2
            + test2_5_1 + test2_5_2 + test2_6_1 + test2_6_2 + test2_7_1 + test2_7_2 + test2_8_1 + test2_8_2
            + test2_9_1 + test2_9_2 + test2_10_1 + test2_10_2
            + test3_1 + test3_2
            + test4_1 + test4_2
            + test5_1 + test5_2
            + test6_1 + test6_2
            + test7_1 + test7_2
            + test8_1_1 + test8_1_2 + test8_2_1 + test8_2_2
            + test9_1 + test9_2
            + test10_1 + test10_2 + test10_3 + test10_4 + test10_5 + test10_6
            + test11_1 + test11_2 + test11_3 + test11_4 + test11_5;


EXPORT OLSAnalytics := summary;
