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
compX2 := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
END;
compX2 makeComposite2(Layout_Cell l, DATASET(Layout_Cell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
END;
A21 := -1.8;
A22 := -4.333;
A23 := 11.13;
B2 := -3.333;
N2 := 100000000;
M2 := 3;
mX2 := tm.RandomPersist(N2, M2, 1.0, 2,'OLSPerformance_1');
sX2 := SORT(mX2, wi_id, x);
gX2 := GROUP(sX2, wi_id, x);
cX2 := ROLLUP(gX2,  GROUP, makeComposite2(LEFT, ROWS(LEFT)));
Layout_Cell makeY2(compX2 X) := TRANSFORM
  SELF.x := X.id;
  SELF.y := 1;
  SELF.wi_id := X.wi;
  SELF.v := A21* X.X1 + A22 * X.X2 + A23 * X.X3 + B2 + Noise;
END;
mY2 := PROJECT(cX2, makeY2(LEFT));
X2 := pbbConverted.MatrixToNF(mX2);
Y2 := pbbConverted.MatrixToNF(my2);
lr2 := LROLS.OLS(X2, Y2);
mod := lr2.GetModel;

EXPORT OLSPerformance := mod;
