/*##############################################################################
    
    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       
       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

// Test that calls various functions and fields from the 
// OLS files and validates them

#ONWARNING(4550, ignore);
#ONWARNING(4531, ignore);

IMPORT LinearRegression AS LR;
IMPORT ML_Core.Types as MLTypes;
IMPORT PBBlas.test.MakeTestMatrix as TM;
IMPORT PBBlas.Types as PBbTypes;
IMPORT PBBlas.Converted as Converted;
IMPORT Python3;

LayoutCell := PBbTypes.Layout_Cell;
LayoutModel := MLTypes.Layout_Model;
NumericField := MLTypes.NumericField;

two31 := POWER(2, 31); 

// Max allowable error
Epsilon := .000001;

REAL Noise(maxv=.1) := FUNCTION
  out := ((RANDOM()-two31)%1000000)/(1000000/maxv);
  return out;
END;

CompX := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
END;

CompX makeComposite(LayoutCell l, DATASET(LayoutCell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
END;

Slope1A := -1.8;
Slope1B := -.333;
Slope1C  := 1.13;
Intercept1 := -3.333;

Slope2A := .0013;
Slope2B := -.123;
Slope2C := .00015;
Intercept2:= -5.01;

N := 50;
M := 3;

MatrixX := TM.RandomMatrix(N, M, 1.0, 2);

X := Converted.MatrixToNF(MatrixX);

GroupedX := GROUP(MatrixX, x, ALL);
CompositedX := ROLLUP(GroupedX,  GROUP, makeComposite(LEFT, ROWS(LEFT)));

LayoutCell makeY(compX X, UNSIGNED c) := TRANSFORM
  SELF.x := X.id;
  SELF.y := IF(c=1, 1, 3);  // Make Y1 and Y3.  Y2 will be a copy of Y1,
  SELF.wi_id := X.wi;
  v1 := Slope1A * X.X1 + Slope2B * X.X2 + Slope1C * X.X3 + Intercept1 + Noise(1);
  v2 := Slope2A * X.X1 + Slope2B * X.X2 + Slope2C * X.X3 + Intercept2 + Noise(10000);
  SELF.v := IF(c=1, v1, v2);
END;

MatrixY := NORMALIZE(CompositedX, 2, makeY(LEFT, COUNTER));

Y := Converted.MatrixToNF(MatrixY);

OLS := LR.OLS(X, Y);

Model := OLS.GetModel();
Betas := OLS.Betas(Model);

// Ensures that each Betas swaps the id and number from model and maintains the same values
BOOLEAN BetasIsAccurate(DATASET(LayoutModel) mod, DATASET(NumericField) bets) := EMBED(Python3)
  for (row, beta) in zip(mod, bets):
    if row.value != beta.value:
      return False 
    if row.id != beta.number:
      return False
    if row.number != beta.id:
      return False

  return True
ENDEMBED;

OUTPUT(IF(BetasIsAccurate(Model, Betas), 'Pass', 'Error with Betas present'), NAMED('Test1'));

RSqrd := OLS.RSquared;

OUTPUT(IF(ROUND(Rsqrd[1].rsquared) = 0, 'Pass', 'Value should lean towards a non-linear fit'), NAMED('Test2A'));
OUTPUT(IF(ROUND(Rsqrd[2].rsquared) = 1, 'Pass', 'Value should lean towards a linear fit'), NAMED('Test2B'));

ConfInt := OLS.ConfInt(50);

InvalidCI := ConfInt(lowerint > upperint);

OUTPUT(IF(NOT EXISTS(InvalidCI), 'Pass', 'Lower intervals should always be less than upper intervals'), NAMED('Test3'));
OUTPUT(InvalidCI, NAMED('InvalidCI'));

Anova := OLS.Anova[1];

// Comparision helper function
STRING GetResult(INTEGER Expected, INTEGER Result) := FUNCTION
      RETURN IF(Expected = Result, 'Pass', 'Expected: ' + Expected + ' Result: ' + Result);
END;

// Test df values of Anova
ExpectedTotDF := 49;
OUTPUT(GetResult(ExpectedTotDF, Anova.Total_Df), NAMED('Test4A'));

ExpectedModDf := 3;
OUTPUT(GetResult(ExpectedModDf, Anova.Model_Df), NAMED('Test4B'));

ExpectedErrDf := 46;
OUTPUT(GetResult(ExpectedErrDf, Anova.Error_Df), NAMED('Test4C'));

DB := OLS.DistributionBase();

// As a result of the Density function returning 0 each P value should be 0.
// If it is needed for the Density function to be updated,
// this test will fail and should be updated as well

CV := DB.CumulativeV();

// All rows have P as 0
Pass := COUNT(CV(P = 0)) = COUNT(CV);
OUTPUT(IF(Pass, 'Pass', 'All rows do not have a P value of 0'), NAMED('Test5'));

// Test return values of the Cumulative function

OUTPUT(GetResult(1, DB.Cumulative(DB.High)), NAMED('Test6A'));
OUTPUT(GetResult(0, DB.Cumulative(0.5)), NAMED('Test6B'));
OUTPUT(GetResult(0, DB.Cumulative(5.7)), NAMED('Test6C'));
OUTPUT(GetResult(0, DB.Cumulative(DB.Low)), NAMED('Test6D'));
