import numpy as np;
from Contingency_Table import contingency_table;

class invariance(object):
    def __init__(self, measure):
        self.table1 = contingency_table(np.array([4,2,3,4]));
        self.table2 = contingency_table(np.array([600,250,300,500]));
        self.measure = measure;
    
    def O5(self, k1 = 1, k2 = 5):
        # O(M + C)
        table1 = contingency_table(self.table1.table  + np.array([0,0,0,k1]));
        table2 = contingency_table(self.table2.table  + np.array([0,0,0,k2]));
        eval_str1 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table1.' + self.measure + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table2.' + self.measure + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));
        
    def O4(self):
        # O(SMS)
        table1 = contingency_table(self.table1.table[np.array([3,2,1,0])]);
        table2 = contingency_table(self.table2.table[np.array([3,2,1,0])]);
        eval_str1 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table1.' + self.measure + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table2.' + self.measure + '(),4)';
        return (eval(eval_str1) and eval(eval_str2));
        
    def O3(self):
        # O(SM)
        table1 = contingency_table(self.table1.table[np.array([2,3,0,1])]);
        table2 = contingency_table(self.table2.table[np.array([2,3,0,1])]);
        eval_str1 = 'np.around(self.table1.' + self.measure + '(),4) == -1 * np.around(table1.' + self.measure + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure + '(),4) == -1 * np.around(table2.' + self.measure + '(),4)';

        # O(MS)
        table3 = contingency_table(self.table1.table[np.array([1,0,3,2])]);
        table4 = contingency_table(self.table2.table[np.array([1,0,3,2])]);
        eval_str3 = 'np.around(self.table1.' + self.measure + '(),4) == -1 * np.around(table3.' + self.measure + '(),4)';
        eval_str4 = 'np.around(self.table2.' + self.measure + '(),4) == -1 * np.around(table4.' + self.measure + '(),4)';

        return (eval(eval_str1) and eval(eval_str2) and eval(eval_str3) and eval(eval_str4));

    
    def O2(self):
        # O(RM) - multiply by [k1, 0; 0, k2] on the left
        #k1 = 2, k2 = 3
        table1 = contingency_table(self.table1.table * np.array([2,2,3,3]));
        #k1 = 3, k2 = 5
        table2 = contingency_table(self.table2.table * np.array([3,3,5,5]));
        eval_str1 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table1.' + self.measure + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table2.' + self.measure + '(),4)';

        # O(MC) - multiply by [k1, 0; 0, k2] on the right
        #k1 = 2, k2 = 3
        table3 = contingency_table(self.table1.table * np.array([2,3,2,3]));
        #k1 = 3, k2 = 5
        table4 = contingency_table(self.table2.table * np.array([3,5,3,5]));
        eval_str3 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table3.' + self.measure + '(),4)';
        eval_str4 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table4.' + self.measure + '(),4)';
        
        return (eval(eval_str1) and eval(eval_str2) and eval(eval_str3) and eval(eval_str4));

    
    def O1(self):
        # O(M_transpose)
        table1 = contingency_table(self.table1.table[np.array([0,2,1,3])]);
        table2 = contingency_table(self.table2.table[np.array([0,2,1,3])]);
        eval_str1 = 'self.table1.' + self.measure + '() == table1.' + self.measure + '()';
        eval_str2 = 'self.table2.' + self.measure + '() == table2.' + self.measure + '()';
        return (eval(eval_str1) and eval(eval_str2));
    
    def P2(self):
        # 
        #k = 1
        table1 = contingency_table(self.table1.table + np.array([1,-1,-1,1]));
        #k = 50
        table2 = contingency_table(self.table2.table + np.array([50,-50,-50,50]));
        eval_str1 = 'self.table1.' + self.measure + '() < table1.' + self.measure + '()';
        eval_str2 = 'self.table2.' + self.measure + '() < table2.' + self.measure + '()';

        return (eval(eval_str1) and eval(eval_str2));
    
    def P3(self):
        # P_ab and P_b unchanged, P_a increased
        #k = 1
        k = 1;
        table1 = contingency_table(self.table1.table + np.array([0,k,0,-k]));
        #k = 50
        k = 50;
        table2 = contingency_table(self.table2.table + np.array([0,k,0,-k]));
        eval_str1 = 'self.table1.' + self.measure + '() > table1.' + self.measure + '()';
        eval_str2 = 'self.table2.' + self.measure + '() > table2.' + self.measure + '()';

        # P_ab and P_a unchanged, P_b increased        
        #k = 1
        k = 1;
        table3 = contingency_table(self.table1.table + np.array([0,0,k,-k]));
        #k = 50
        k = 50;
        table4 = contingency_table(self.table2.table + np.array([0,0,k,-k]));
        eval_str3 = 'self.table1.' + self.measure + '() > table3.' + self.measure + '()';
        eval_str4 = 'self.table2.' + self.measure + '() > table4.' + self.measure + '()';
        return (eval(eval_str1) and eval(eval_str2) and eval(eval_str3) and eval(eval_str4));

    def uniform_scaling(self):
        # O(kM) - multiply by [k, 0; 0, k] on the left
        #k = 3
        table1 = contingency_table(self.table1.table * np.array([3,3,3,3]));
        #k = 5
        table2 = contingency_table(self.table2.table * np.array([5,5,5,5]));
        eval_str1 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table1.' + self.measure + '(),4)';
        eval_str2 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table2.' + self.measure + '(),4)';

        # # O(MC) - multiply by [k1, 0; 0, k2] on the right
        # #k1 = 2, k2 = 3
        # table3 = contingency_table(self.table1.table * np.array([2,3,2,3]));
        # #k1 = 3, k2 = 5
        # table4 = contingency_table(self.table2.table * np.array([3,5,3,5]));
        # eval_str3 = 'np.around(self.table1.' + self.measure + '(),4) == np.around(table3.' + self.measure + '(),4)';
        # eval_str4 = 'np.around(self.table2.' + self.measure + '(),4) == np.around(table4.' + self.measure + '(),4)';
        
        return (eval(eval_str1) and eval(eval_str2));