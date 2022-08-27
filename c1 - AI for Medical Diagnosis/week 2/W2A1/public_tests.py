import numpy as np
import pandas as pd
from IPython.display import display
from test_utils import *

### ex1
def get_tp_tn_fp_fn_test(target_1, target_2, target_3, target_4):
    threshold = 0.5
    
    df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                       'preds_test': [0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0],
                       'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                      })
    
    y_test = df['y_test']
    preds_test = df['preds_test']
    
    display(df)
    print(f"""Your functions calcualted: 
    TP: {target_1(y_test, preds_test, threshold)}
    TN: {target_2(y_test, preds_test, threshold)}
    FP: {target_3(y_test, preds_test, threshold)}
    FN: {target_4(y_test, preds_test, threshold)}
    """)
    
    expected_output_1 = np.int64(sum(df['category'] == 'TP'))
    expected_output_2 = np.int64(sum(df['category'] == 'TN'))
    expected_output_3 = np.int64(sum(df['category'] == 'FP'))
    expected_output_4 = np.int64(sum(df['category'] == 'FN'))
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in true_positives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in true_positives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in true_positives"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in true_negatives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in true_negatives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in true_negatives"
        }
    ]
    
    multiple_test(test_cases_2, target_2)
    
    test_cases_3 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Data-type mismatch in false_positives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Wrong shape in false_positives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_3,
            "error": "Wrong output in false_positives"
        }
    ]
    
    multiple_test(test_cases_3, target_3)
    
    test_cases_4 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Data-type mismatch in false_negatives"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Wrong shape in false_negatives"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_4,
            "error": "Wrong output in false_negatives"
        }
    ]
    
    multiple_test(test_cases_4, target_4)

### ex2
def get_accuracy_test(target):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t  ", y_test)
    print("Test Predictions: ", preds_test)
    print("Threshold:\t  ", threshold)
    print("Computed Accuracy:", target(y_test, preds_test, threshold), "\n")
    
    expected_output = np.float64(0.6)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)

### ex3
def get_prevalence_test(target):
    y_test = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    
    print("Test Case:\n")
    print("Test Labels:\t     ", y_test)
    print("Computed Prevalence: ", target(y_test), "\n")
    
    expected_output = np.float64(0.4)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [y_test],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)

### ex4
def get_sensitivity_specificity_test(target_1, target_2):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t      ", y_test)
    print("Test Predictions:     ", y_test)
    print("Threshold:\t      ", threshold)
    print("Computed Sensitivity: ", target_1(y_test, preds_test, threshold))
    print("Computed Specificity: ", target_2(y_test, preds_test, threshold), "\n")
    
    expected_output_1 = np.float64(0.6666666666666666)
    expected_output_2 = np.float64(0.5)
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in get_sensitivity."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in get_sensitivity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in get_sensitivity"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in get_specificity"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in get_specificity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in get_specificity"
        }
    ]
    
    multiple_test(test_cases_2, target_2)

### ex5
def get_ppv_npv_test(target_1, target_2):
    y_test = np.array([1, 0, 0, 1, 1])
    preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
    threshold = 0.5
    
    print("Test Case:\n")
    print("Test Labels:\t  ", y_test)
    print("Test Predictions: ", y_test)
    print("Threshold:\t  ", threshold)
    print("Computed PPV:\t  ", target_1(y_test, preds_test, threshold))
    print("Computed NPV:\t  ", target_2(y_test, preds_test, threshold),"\n")
    
    expected_output_1 = np.float64(0.6666666666666666)
    expected_output_2 = np.float64(0.5)
    
    test_cases_1 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Data-type mismatch in get_ppv."
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong shape in get_ppv"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_1,
            "error": "Wrong output in get_ppv"
        }
    ]
    
    multiple_test(test_cases_1, target_1)
    
    test_cases_2 = [
        {
            "name":"datatype_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Data-type mismatch in get_specificity"
        },
        {
            "name": "shape_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong shape in get_specificity"
        },
        {
            "name": "equation_output_check",
            "input": [y_test, preds_test, threshold],
            "expected": expected_output_2,
            "error": "Wrong output in get_specificity"
        }
    ]
    
    multiple_test(test_cases_2, target_2)


