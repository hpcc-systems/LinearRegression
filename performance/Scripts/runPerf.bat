SETLOCAL
ECHO OFF
REM runPerf.bat <clustersize>
REM Run each of the performance tests for each of the test sized (small, med, large)
SET host=52.9.80.51
SET lib="\Users\devrog01\source\LinearRegression"
ECHO cluster size = %1

REM test_size = 1 is small, test_size = 2 is medium, test_size = 3 is large
REM First run the Prep program to create the files.  Then run the test program
REM to get the performance timing for each size.
REM Run small first to avoid wasted time if there is a problem.
REM Set pathPrefix to the folder containing PBblas
SET pathPrefix="\Users\devrog01\source"
SET lib=%pathPrefix%\LinearRegression

REM OLSPerf -- Run OLS GetModel
SET progName=OLSPerf
SET prepPath=%pathPrefix%\LinearRegression\performance\%progName%Prep.ecl
SET progPath=%pathPrefix%\LinearRegression\performance\%progName%.ecl
ecl run mythor %prepPath% -s %host% --port 8010 -I%lib% -v --name=%progName%Prep --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=1 -I%lib% -v --name=%progName%_S_%1 --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=2 -I%lib% -v --name=%progName%_M_%1 --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=3 -I%lib% -v -fthorConnectTimeout=10000000 --name=%progName%_L_%1 --wait=10000000

REM OLSPerfMyr -- Run OLS GetModel as Myriad
SET progName=OLSPerfMyr
SET prepPath=%pathPrefix%\LinearRegression\performance\%progName%Prep.ecl
SET progPath=%pathPrefix%\LinearRegression\performance\%progName%.ecl
ecl run mythor %prepPath% -s %host% --port 8010 -I%lib% -v --name=%progName%Prep --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=1 -I%lib% -v --name=%progName%_S_%1 --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=2 -I%lib% -v --name=%progName%_M_%1 --wait=10000000
ecl run mythor %progPath% -s %host% --port 8010 -Xtest_size=3 -I%lib% -v -fthorConnectTimeout=10000000 --name=%progName%_L_%1 --wait=10000000

ECHO 
ENDLOCAL
