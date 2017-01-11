/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $.^ as LROLS;
IMPORT ML_Core;
IMPORT ML_Core.Types as mlTypes;
IMPORT PBblas;
IMPORT PBBlas.test.MakeTestMatrix as tm;
IMPORT PBBlas.Types as pbbTypes;
IMPORT PBBlas.Converted as pbbConverted;
Layout_Cell := pbbTypes.Layout_Cell;
NumericField := mlTypes.NumericField;
two31 := POWER(2, 31);
REAL Noise := FUNCTION
  out := ((RANDOM()-two31)%1000)/10000;
  return out;
END;

test_rslt := RECORD
  STRING32 TestName;
  SET OF REAL X;
  REAL Y;
  REAL projY;
  REAL diff;
  REAL pctErr;
END;


// TEST2 -- Multiple Regression
compX := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
  REAL X4;
  REAL X5;
  REAL X6;
  REAL X7;
  REAL X8;
  REAL X9;
  REAL X10;
END;

compX makeComposite(Layout_Cell l, DATASET(Layout_Cell) r, UNSIGNED test_size) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
  SELF.X4 := IF(test_size > 1, r(y=4)[1].v, 0);
  SELF.X5 := IF(test_size > 1, r(y=5)[1].v, 0);
  SELF.X6 := IF(test_size > 1, r(y=6)[1].v, 0);
  SELF.X7 := IF(test_size > 2, r(y=7)[1].v, 0);
  SELF.X8 := IF(test_size > 2, r(y=8)[1].v, 0);
  SELF.X9 := IF(test_size > 2, r(y=9)[1].v, 0);
  SELF.X10 := IF(test_size > 2, r(y=10)[1].v, 0);
END;

A1 := -1.8;
A2 := -4.333;
A3 := 11.13;
A4 := 1.067;
A5 := -.541;
A6 := -.233;
A7 := 2.987;
A8 := 1.0;
A9 := -1.0;
A10 := 2.123;

B := -3.333;

N_s := 100000;
N_m := 10000000;
N_l := 100000000;
M_s := 3; // If any M values are changed, must also change makeComposite above to match.
M_m := 6;
M_l := 10;

Layout_Cell makeY(compX X) := TRANSFORM
  SELF.x := X.id;
  SELF.y := 1;
  SELF.wi_id := X.wi;
  SELF.v := A1* X.X1
          + A2 * X.X2
          + A3 * X.X3
          + A4 * X.X4
          + A5 * X.X5
          + A6 * X.X6
          + A7 * X.X7
          + A8 * X.X8
          + A9 * X.X9
          + A10 * X.X10
          + B + Noise;
END;

// Make the 'small' data
mX_s := tm.RandomMatrix(N_s, M_s, 1.0, 1);
sX_s := SORT(mX_s, wi_id, x);
gX_s := GROUP(sX_s, wi_id, x);
cX_s := ROLLUP(gX_s,  GROUP, makeComposite(LEFT, ROWS(LEFT), 1));
mY_s := PROJECT(cX_s, makeY(LEFT));
X_s := pbbConverted.MatrixToNF(mX_s);
Y_s := pbbConverted.MatrixToNF(my_s);
OUTPUT(X_s,, 'OLSPerf_X_s.dat', OVERWRITE);
OUTPUT(Y_s,, 'OLSPerf_Y_s.dat', OVERWRITE);

// Make the 'medium' data
mX_m := tm.RandomMatrix(N_m, M_m, 1.0, 1);
sX_m := SORT(mX_m, wi_id, x);
gX_m := GROUP(sX_m, wi_id, x);
cX_m := ROLLUP(gX_m,  GROUP, makeComposite(LEFT, ROWS(LEFT), 2));
mY_m := PROJECT(cX_m, makeY(LEFT));
X_m := pbbConverted.MatrixToNF(mX_m);
Y_m := pbbConverted.MatrixToNF(my_m);
OUTPUT(X_m,, 'OLSPerf_X_m.dat', OVERWRITE);
OUTPUT(Y_m,, 'OLSPerf_Y_m.dat', OVERWRITE);

// Make the 'large' data
mX_l := tm.RandomMatrix(N_l, M_l, 1.0, 1);
sX_l := SORT(mX_l, wi_id, x);
gX_l := GROUP(sX_l, wi_id, x);
cX_l := ROLLUP(gX_l,  GROUP, makeComposite(LEFT, ROWS(LEFT), 3));
mY_l := PROJECT(cX_l, makeY(LEFT));
X_l := pbbConverted.MatrixToNF(mX_l);
Y_l := pbbConverted.MatrixToNF(my_l);
OUTPUT(X_l,, 'OLSPerf_X_l.dat', OVERWRITE);
OUTPUT(Y_l,, 'OLSPerf_Y_l.dat', OVERWRITE);
