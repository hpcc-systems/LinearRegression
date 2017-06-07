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

compX makeComposite(Layout_Cell l, DATASET(Layout_Cell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
  SELF.X4 := r(y=4)[1].v;
  SELF.X5 := r(y=5)[1].v;
  SELF.X6 := r(y=6)[1].v;
  SELF.X7 := r(y=7)[1].v;
  SELF.X8 := r(y=8)[1].v;
  SELF.X9 := r(y=9)[1].v;
  SELF.X10 := r(y=10)[1].v;
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

N := 100000;
M := 10;
Myriad_count_s := 50;
Myriad_count_m := 100;
Myriad_count_l := 500;

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

// Make the basic small test data
mX := tm.RandomMatrix(N, M, 1.0, 1);
sX := SORT(mX, wi_id, x);
gX := GROUP(sX, wi_id, x);
cX := ROLLUP(gX,  GROUP, makeComposite(LEFT, ROWS(LEFT)));
mY := PROJECT(cX, makeY(LEFT));
X := pbbConverted.MatrixToNF(mX);
Y := pbbConverted.MatrixToNF(mY);
// Now copy it multiple times to make the small, medium, and large datasets
NumericField makeMyriad(NumericField nf, UNSIGNED c) := TRANSFORM
  SELF.wi := c;
  SELF    := nf;
END;

X_s := NORMALIZE(X, Myriad_count_s, makeMyriad(LEFT, COUNTER));
Y_s := NORMALIZE(Y, Myriad_count_s, makeMyriad(LEFT, COUNTER));

X_m := NORMALIZE(X, Myriad_count_m, makeMyriad(LEFT, COUNTER));
Y_m := NORMALIZE(Y, Myriad_count_m, makeMyriad(LEFT, COUNTER));

X_l := NORMALIZE(X, Myriad_count_l, makeMyriad(LEFT, COUNTER));
Y_l := NORMALIZE(Y, Myriad_count_l, makeMyriad(LEFT, COUNTER));


OUTPUT(X_s,, 'OLSPerfMyr_X_s.dat', OVERWRITE);
OUTPUT(Y_s,, 'OLSPerfMyr_Y_s.dat', OVERWRITE);
OUTPUT(X_m,, 'OLSPerfMyr_X_m.dat', OVERWRITE);
OUTPUT(Y_m,, 'OLSPerfMyr_Y_m.dat', OVERWRITE);
OUTPUT(X_l,, 'OLSPerfMyr_X_l.dat', OVERWRITE);
OUTPUT(Y_l,, 'OLSPerfMyr_Y_l.dat', OVERWRITE);
