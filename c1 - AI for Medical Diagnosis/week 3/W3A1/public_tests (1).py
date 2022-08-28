import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from keras import backend
from test_case import *
from IPython.display import display
np.random.seed(3)

### ex1
def get_sub_volume_test(target):
    np.random.seed(3)
    image, label = get_sub_volume_test_case()

    print("Image:")
    for k in range(3):
        print(f"z = {k}")
        print(image[:, :, k, 0])

    print("\n")
    print("Label:")
    for k in range(3):
        print(f"z = {k}")
        print(label[:, :, k])
        
    print("\033[1m\nExtracting (2, 2, 2) sub-volume\n\033[0m")
    
    orig_x = 4
    orig_y = 4
    orig_z = 3
    output_x = 2
    output_y = 2
    output_z = 2
    num_classes = 3
    
    expected_output = (np.array([[[[1., 2.],
                                   [2., 4.]],
                                  [[2., 4.],
                                   [4., 8.]]]]), 
                       np.array([[[[1., 0.],
                                   [1., 0.]],
                                  [[1., 0.],
                                   [1., 0.]]],
                                 [[[0., 1.],
                                   [0., 1.]],
                                  [[0., 1.],
                                   [0., 1.]]]], dtype=np.float32))
    
    test_cases = [
         {
             "name":"datatype_check",
             "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
             "expected": expected_output,
             "error": "Data-type mismatch."
         },
        {
            "name": "shape_check",
            "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [image, label, orig_x, orig_y, orig_z, output_x, output_y, output_z, num_classes],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    learner_func_sample_image, learner_func_sample_label = multiple_test_get_sub_volume(test_cases, target)
    
    print("\033[0m\nSampled Image:")
    for k in range(2):
        print("z = " + str(k))
        print(learner_func_sample_image[0, :, :, k])
        
    print("\nSampled Label:")
    for c in range(2):
        print("class = " + str(c))
        for k in range(2):
            print("z = " + str(k))
            print(learner_func_sample_label[c, :, :, k])
    


##############################################        
### ex2
def standardize_test(target, X):
    X_norm = target(X)
    
    def return_x_norm_value(X_norm): 
        return X_norm[0,:,:,0].std()
    
    print("stddv for X_norm[0, :, :, 0]: ", return_x_norm_value(X_norm), "\n")
    
    expected_output = np.float64(0.9999999999999999)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [X_norm],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, return_x_norm_value)
    


##############################################        
### ex3
def single_class_dice_coefficient_test(target, epsilon, sess):
    pred_1, label_1, pred_2, label_2 = single_class_dice_coefficient_test_case(sess)
        
    expected_output_1 = np.float64(0.6)
    expected_output_2 = np.float64(0.8333333333333334) 
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[:, :, 0])
    print("\nLabel:\n")
    print(label_1[:, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nDice coefficient: ", dc_1.eval(session=sess), "\n\n----------------------\n")
        
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[:, :, 0])
    print("\nLabel:\n")
    print(label_2[:, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nDice coefficient: ", dc_2.eval(session=sess), "\n")
        
    axis = (0, 1, 2)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        }
    ]
    
    multiple_test_dice(test_cases, target, sess)
        

##############################################    
### ex4
def dice_coefficient_test(target, epsilon, sess):
    pred_1, label_1, pred_2, label_2, pred_3, label_3 = dice_coefficient_test_case(sess)
        
    expected_output_1 = np.float64(0.6)
    expected_output_2 = np.float64(0.8333333333333334)
    expected_output_3 = np.float64(0.7166666666666667) 
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nDice coefficient: ", dc_1.eval(session=sess), "\n\n----------------------\n")
        
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nDice coefficient: ", dc_2.eval(session=sess), "\n\n----------------------\n")
        
    print("Test Case 3:\n")
    print("Pred:\n")
    print("class = 0")
    print(pred_3[0, :, :, 0], "\n")
    print("class = 1")
    print(pred_3[1, :, :, 0], "\n")
    print("Label:\n")
    print("class = 0")
    print(label_3[0, :, :, 0], "\n")
    print("class = 1")
    print(label_3[1, :, :, 0], "\n")

    dc_3 = target(label_3, pred_3, epsilon=epsilon)
    print("Dice coefficient: ", dc_3.eval(session=sess), "\n")
        
    axis = (1, 2, 3)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Data-type mismatch for Test Case 3"
        },
        {
            "name": "shape_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Wrong shape for Test Case 3"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3. One possible reason for error: make sure epsilon = 1"
        }
    ]

    multiple_test_dice(test_cases, target, sess)
        
##############################################         
### ex5
def soft_dice_loss_test(target, epsilon, sess):
    pred_1, label_1, pred_2, label_2, pred_3, label_3, pred_4, label_4, pred_5, label_5, pred_6, label_6 = soft_dice_loss_test_case(sess)
    
    expected_output_1 = np.float64(0.4)
    expected_output_2 = np.float64(0.4285714285714286)
    expected_output_3 = np.float64(0.16666666666666663)
    expected_output_4 = np.float64(0.006024096385542355)
    expected_output_5 = np.float64(0.21729776247848553)
    expected_output_5 = np.float64(0.21729776247848553)
    expected_output_6 = np.float64(0.4375)
        
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])

    dc_1= target(pred_1, label_1, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_1.eval(session=sess), "\n\n----------------------\n")
    
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])

    dc_2= target(pred_2, label_2, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_2.eval(session=sess), "\n\n----------------------\n")
    
    print("Test Case 3:\n")
    print("Pred:\n")
    print(pred_3[0, :, :, 0])
    print("\nLabel:\n")
    print(label_3[0, :, :, 0])

    dc_3= target(pred_3, label_3, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_3.eval(session=sess), "\n\n----------------------\n")
    
    print("Test Case 4:\n")
    print("Pred:\n")
    print(pred_4[0, :, :, 0])
    print("\nLabel:\n")
    print(label_4[0, :, :, 0])

    dc_4= target(pred_4, label_4, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_4.eval(session=sess), "\n\n----------------------\n")
    
    print("Test Case 5:\n")
    print("Pred:\n")
    print("class = 0")
    print(pred_5[0, :, :, 0], "\n")
    print("class = 1")
    print(pred_5[1, :, :, 0], "\n")
    print("Label:\n")
    print("class = 0")
    print(label_5[0, :, :, 0], "\n")
    print("class = 1")
    print(label_5[1, :, :, 0], "\n")

    dc_5 = target(label_5, pred_5, epsilon=epsilon)
    print("\nSoft Dice Loss: ", dc_5.eval(session=sess), "\n\n----------------------\n")
    
    print("Test Case 6:\n")
    dc_6 = target(label_6, pred_6, epsilon=epsilon)
    print("Soft Dice Loss: ", dc_6.eval(session=sess), "\n")
        
    axis = (1, 2, 3)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, axis, epsilon],
            "expected": expected_output_1,
             "error": "Wrong output for Test Case 1. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, axis, epsilon],
            "expected": expected_output_2,
             "error": "Wrong output for Test Case 2. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, axis, epsilon],
            "expected": expected_output_3,
             "error": "Wrong output for Test Case 3. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_4, label_4, axis, epsilon],
            "expected": expected_output_4,
             "error": "Wrong output for Test Case 4. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
            "error": "Data-type mismatch for Test Case 5."
        },
        {
            "name": "shape_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
            "error": "Wrong shape for Test Case 5."
        },
        {
            "name": "equation_output_check",
            "input": [pred_5, label_5, axis, epsilon],
            "expected": expected_output_5,
             "error": "Wrong output for Test Case 5. One possible reason for error: make sure epsilon = 1"
        },
        {
            "name":"datatype_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
            "error": "Data-type mismatch for Test Case 6."
        },
        {
            "name": "shape_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
            "error": "Wrong shape for Test Case 6."
        },
        {
            "name": "equation_output_check",
            "input": [pred_6, label_6, axis, epsilon],
            "expected": expected_output_6,
             "error": "Wrong output for Test Case 6. One possible reason for error: make sure epsilon = 1"
        },        
    ]
   
    multiple_test_dice(test_cases, target, sess)
    
##############################################    
### ex6    
def compute_class_sens_spec_test(target):
    pred_1, label_1, pred_2, label_2, df = compute_class_sens_spec_test_case()
    
    expected_output_1 = np.array((0.5, 0.5))
    expected_output_2 = np.array(((0.6666666666666666, 1.0)))
    expected_output_3 = np.array(((0.2857142857142857, 0.42857142857142855)))
    
    
    sensitivity_1, specificity_1 = target(pred_1, label_1, 0)
    print("Test Case 1:\n")
    print("Pred:\n")
    print(pred_1[0, :, :, 0])
    print("\nLabel:\n")
    print(label_1[0, :, :, 0])
    print("\nSensitivity: ", sensitivity_1)
    print("Specificity: ", specificity_1, "\n\n----------------------\n")
    
    sensitivity_2, specificity_2 = target(pred_2, label_2, 0)
    print("Test Case 2:\n")
    print("Pred:\n")
    print(pred_2[0, :, :, 0])
    print("\nLabel:\n")
    print(label_2[0, :, :, 0])
    print("\nSensitivity: ", sensitivity_2)
    print("Specificity: ", specificity_2, "\n\n----------------------\n")
    
    print("Test Case 3:")
    display(df)
    pred_3 = np.array( [df['preds_test']])
    label_3 = np.array( [df['y_test']])
    sensitivity_3, specificity_3 = target(pred_3, label_3, 0)
    print("\nSensitivity: ", sensitivity_3)
    print("Specificity: ", specificity_3, "\n")
    
    test_cases = [
        {
             "name":"datatype_check",
             "input": [pred_1, label_1, 0],
             "expected": expected_output_1,
             "error": "Data-type mismatch for Test Case 1"
         },
        {
            "name": "shape_check",
            "input": [pred_1, label_1, 0],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_1, label_1, 0],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1"
        },
        {
            "name": "equation_output_check",
            "input": [pred_2, label_2, 0],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2"
        },
        {
            "name": "equation_output_check",
            "input": [pred_3, label_3, 0],
            "expected": expected_output_3,
            "error": "Wrong output for Test Case 3"
        }
    ]
    
    multiple_test(test_cases, target)
    


