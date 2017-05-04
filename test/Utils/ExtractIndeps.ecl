// Extract the independent variables from a test dataset.  The independent variables
// are any variables not named 'class'.  Returns a TABLE of records with only the
// independent variables.
EXPORT ExtractIndeps(Resdata, dOut) := MACRO
    IMPORT Std;
    LOADXML('<xml/>');
    #EXPORTXML(fields, RECORDOF(Resdata));

    #DECLARE(OutStr);
    #SET(OutStr,'{');

    #FOR(fields)
      #FOR(Field)
        #IF ((%'{@name}'% <> 'class')AND (%'{@name}'% <> 'select_number'))
          #APPEND(OutStr, %'{@name}'%);
          #APPEND(OutStr, ',');
        #END
      #END
    #END
    #APPEND(OutStr, '}');
                // %'OutStr'%
    dOut := TABLE(Resdata, #EXPAND(%'OutStr'%));
ENDMACRO;