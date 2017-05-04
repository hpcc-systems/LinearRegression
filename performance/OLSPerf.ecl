/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
/**
  * Performance test for OLS Get Model.  Performs a getModel (small medium or large).
  * Data is staged by OLSPerfPrep.ecl
  * which should always be run before this test.  Test size is controlled by
  * passing test_size as a STORED parameter e.g. via eclcc -Xtest_size=<1, 2, or 3>.
  * See test_size below.
  */
IMPORT $.^ as LROLS;
IMPORT ML_Core;
IMPORT ML_Core.Types as mlTypes;
IMPORT PBblas;
IMPORT PBBlas.test.MakeTestMatrix as tm;
IMPORT PBBlas.Types as pbbTypes;
IMPORT PBBlas.Converted as pbbConverted;
Layout_Cell := pbbTypes.Layout_Cell;
NumericField := mlTypes.NumericField;

// Test size is 1 (small), 2 (medium) or 3 (large)
UNSIGNED test_size := 1 : STORED('test_size'); // default is "small"

X_s := DATASET('OLSPerf_X_s.dat', NumericField, FLAT);
Y_s := DATASET('OLSPerf_Y_s.dat', NumericField, FLAT);
X_m := DATASET('OLSPerf_X_m.dat', NumericField, FLAT);
Y_m := DATASET('OLSPerf_Y_m.dat', NumericField, FLAT);
X_l := DATASET('OLSPerf_X_l.dat', NumericField, FLAT);
Y_l := DATASET('OLSPerf_Y_l.dat', NumericField, FLAT);

X := MAP(test_size = 1 => X_s, test_size = 2 => X_m, X_l);
Y := MAP(test_size = 1 => Y_s, test_size = 2 => Y_m, Y_l);

lr := LROLS.OLS(X, Y);
mod := lr.GetModel;


EXPORT OLSPerf := mod;
