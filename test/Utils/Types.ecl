EXPORT Types := MODULE
    EXPORT result_rec := RECORD
        INTEGER dataset_id; // This is based on how order the datasets are ordered in Test*.ecl
        REAL scores;
    END;
    EXPORT t_test_distribution_table_rec := RECORD
        INTEGER degree_of_freedom;
        REAL t_value;
    END;
    EXPORT dataset_record := RECORD
        INTEGER dataset_id;
        STRING dataset_name;
        REAL ecl_performance;
        REAL scikit_learn_performance;
        STRING status;
    END;
END;