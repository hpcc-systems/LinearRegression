EXPORT GetNumberofFields(Resdata,numF) := MACRO
    IMPORT Std;
    LOADXML('<xml/>');
    #EXPORTXML(fields, RECORDOF(Resdata));
                #DECLARE(fnum);
                #SET(fnum, 0);
    #FOR(fields)
        #FOR(Field)
          #IF ((%'{@name}'% <> 'class') AND (%'{@name}'% <> 'id'))
                                                #SET(fnum, %fnum%+1);
          #END
        #END
    #END
                numF := %fnum%;
ENDMACRO;
