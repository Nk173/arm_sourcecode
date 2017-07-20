import numpy as np;
from Contingency_Table import contingency_table;

class invariance_updated(object):
    def __init__(self, measure_name):
        self.table1 = contingency_table(np.array([4,2,3,4]));
        self.table2 = contingency_table(np.array([600,250,300,500]));
        self.measure_name = measure_name;
        
    # null invariance
    def P1(self, k1 = 1, k2 = 5):
        # O(M + C)
        table1 = contingency_table(self.table1.table  + np.array([0,0,0,k1]));
        table2 = contingency_table(self.table2.table  + np.array([0,0,0,k2]));
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # variable permutation
    def P2(self):
        # O(M_transpose)
        table1 = contingency_table(self.table1.table[np.array([0,2,1,3])]);
        table2 = contingency_table(self.table2.table[np.array([0,2,1,3])]);
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # off diagonal permutation
    def P3(self):
        table1 = contingency_table(self.table1.table[np.array([3,1,2,0])]);
        table2 = contingency_table(self.table2.table[np.array([3,1,2,0])]);
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # inversion invariance
    def P4(self):
        # O(SMS)
        table1 = contingency_table(self.table1.table[np.array([3,2,1,0])]);
        table2 = contingency_table(self.table2.table[np.array([3,2,1,0])]);
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # row permutation
    def P5(self):
        # O(SM)
        table1 = contingency_table(self.table1.table[np.array([2,3,0,1])]);
        table2 = contingency_table(self.table2.table[np.array([2,3,0,1])]);
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # column permutation
    def P6(self):
        # O(MS)
        table1 = contingency_table(self.table1.table[np.array([1,0,3,2])]);
        table2 = contingency_table(self.table2.table[np.array([1,0,3,2])]);
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # row scaling
    def P7(self):
        # O(RM) - multiply by [k1, 0; 0, k2] on the left
        #k1 = 2, k2 = 3
        table1 = contingency_table(self.table1.table * np.array([2,2,3,3]));
        #k1 = 3, k2 = 5
        table2 = contingency_table(self.table2.table * np.array([3,3,5,5]));
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # column scaling
    def P8(self):
        # O(MC) - multiply by [k1, 0; 0, k2] on the right
        #k1 = 2, k2 = 3
        table1 = contingency_table(self.table1.table * np.array([2,3,2,3]));
        #k1 = 3, k2 = 5
        table2 = contingency_table(self.table2.table * np.array([3,5,3,5]));
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # f11 + f10 invariance
    def P9(self):
        k1 = 2;
        k2 = 3;
        table1 = contingency_table(self.table1.table  + np.array([0,0,k1,k2]));
        table2 = contingency_table(self.table2.table  + np.array([0,0,k1,k2]));
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));

    # f01 + f11 invariance
    def P10(self):
        k1 = 2;
        k2 = 3;
        table1 = contingency_table(self.table1.table  + np.array([k1,k2,0,0]));
        table2 = contingency_table(self.table2.table  + np.array([k1,k2,0,0]));
        eval_str1 = 'np.around(self.table1.' + self.measure_name + '(),4) == np.around(table1.' + self.measure_name + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure_name + '(),4) == np.around(table2.' + self.measure_name + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));
        