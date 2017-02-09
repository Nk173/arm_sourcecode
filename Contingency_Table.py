import numpy as np;
from Measures import map_measures_to_indices;

(measures_dict, measures_arr) = map_measures_to_indices();


class contingency_table(object):
    def __init__(self, table):
        if np.min(table) < 0:
            raise ValueError('frequency of the matrix cannot be negative');
        self.table = table;
        self.scores = np.zeros(len(measures_dict));
        self.measures = measures_dict;
        self.compute_probabilities();
    
    def compute_probabilities(self):
        self.f11 = self.table[0];
        self.f10 = self.table[1];
        self.f01 = self.table[2];
        self.f00 = self.table[3];
        self.N = sum(self.table);
        
        self.P_a = (self.f11 + self.f10)/self.N;
        self.P_b = (self.f11 + self.f01)/self.N;
        self.P_aprime = 1 - self.P_a;
        self.P_bprime = 1 - self.P_b;

        
        self.P_ab = self.f11/self.N;
        self.P_abprime = self.f10/self.N;
        self.P_aprimeb = self.f01/self.N;
        self.P_aprimebprime = self.f00/self.N;

        self.P_agivenb = self.P_ab / self.P_b;
        self.P_bgivena = self.P_ab / self.P_a;
        self.P_bgivenaprime = self.P_aprimeb / self.P_aprime;
        self.P_aprimegivenbprime = self.P_aprimebprime / self.P_bprime;
        self.P_agivenbprime = 1 - self.P_aprimegivenbprime;
        self.P_bprimegivena = 1 - self.P_bgivena;
        self.P_bprimegivenaprime = 1 - self.P_bgivenaprime;
    
    def compute_scores(self):
        self.compute_probabilities();
        self.scores[0] = self.recall();
        self.scores[1] = self.precision();
        self.scores[2] = self.confidence();
        self.scores[3] = self.mutual_information();
        self.scores[4] = self.jaccard();
        self.scores[5] = self.f_measure();
        self.scores[6] = self.odds_ratio();
        self.scores[7] = self.specificity();
        self.scores[8] = self.negative_reliability();
        self.scores[9] = self.sebag_schoenauer();
        self.scores[10] = self.accuracy();
        self.scores[11] = self.support();
        self.scores[12] = self.confidence_causal();
        self.scores[13] = self.lift();
        self.scores[14] = self.ganascia();
        self.scores[15] = self.kulczynsky_1();
        self.scores[16] = self.coverage();
        self.scores[17] = self.prevalence();
        self.scores[18] = self.relative_risk();
        self.scores[19] = self.piatetsky_shapiro();
        self.scores[20] = self.novelty();
        self.scores[21] = self.yules_q();
        self.scores[22] = self.yules_y();
        self.scores[23] = self.cosine();
        self.scores[24] = self.least_contradiction();
        self.scores[25] = self.odd_multiplier();
        self.scores[26] = self.confirm_descriptive();
        self.scores[27] = self.confirm_causal();
        self.scores[28] = self.certainty_factor();
        self.scores[29] = self.loevinger();
        self.scores[30] = self.conviction();
        self.scores[31] = self.information_gain();
        self.scores[32] = self.laplace_correction();
        self.scores[33] = self.klosgen();
        self.scores[34] = self.zhang();
        self.scores[35] = self.normalized_mutual_information();
        self.scores[36] = self.one_way_support();
        self.scores[37] = self.two_way_support();
        self.scores[self.measures['implication_index']] = self.implication_index();
        self.scores[self.measures['gini_index']] = self.gini_index();
        self.scores[self.measures['goodman_kruskal']] = self.goodman_kruskal();
        self.scores[self.measures['leverage']] = self.leverage();
        self.scores[self.measures['kappa']] = self.kappa();
        self.scores[self.measures['putative_causal_dependency']] = self.putative_causal_dependency();
        self.scores[self.measures['example_counterexample_rate']] = self.example_counterexample_rate();
        self.scores[self.measures['confirmed_confidence_causal']] = self.confirmed_confidence_causal();
    
    def recall (self):
        if (self.f11 + self.f01) == 0:
            return np.nan
        else:
            return self.f11 / (self.f11 + self.f01);

    def precision (self):
        if (self.f11 + self.f10) == 0:
            return np.nan;
        else:
            return self.f11/(self.f11 + self.f10);
    
    def confidence (self):
        return self.precision();
    
    def mutual_information(self):
        if self.f11 != 0 and self.f00 != 0 and self.f01 != 0 and self.f10 != 0:
            MI = self.P_ab * np.log2(self.P_ab/(self.P_a * self.P_b))
            + self.P_abprime * np.log2(self.P_abprime/(self.P_a * self.P_bprime))
            + self.P_aprimeb * np.log2(self.P_aprimeb/(self.P_aprime * self.P_b))
            + self.P_aprimebprime * np.log2(self.P_aprimebprime/(self.P_aprime * self.P_bprime));        
            return MI/self.N;
        else:
            return np.nan;

    def jaccard (self):
        if (self.f11 + self.f10 + self.f01) == 0:
            return np.nan;
        else:
            J = self.f11 / (self.f11 + self.f10 + self.f01);
            return J;

    def f_measure (self):
        if self.P_agivenb + self.P_bgivena == 0:
            return np.nan;
        else:
            FM = (2 * self.P_agivenb * self.P_bgivena) / (self.P_agivenb + self.P_bgivena);
            return FM;
    
    def odds_ratio(self):
        OR = (self.f11 * self.f00)/(self.f01 * self.f10);
        return OR;

    def specificity(self):
        den = self.f00 + self.f01;
        if den != 0:
            return self.f00/den;
        else:
            return np.nan;
    
    def negative_reliability(self):
        return self.specificity();

    def sebag_schoenauer(self):
        if self.f10:
            return self.f11/self.f10;
        else:
            return np.nan;
        
    def accuracy(self):
        if self.N:
            return (self.f11 + self.f00)/self.N;
        else:
            return np.nan;

    def support(self):
        return self.f11/self.N;

    def confidence_causal(self):
        CC = (self.f11 / (self.f11 + self.f10)) + (self.f00 / (self.f00 + self.f10));
        return CC/2;

    def lift(self):
        return (self.f11 * self.N)/((self.f11 + self.f10) * (self.f11 + self.f01));

    def ganascia(self):
        G = 2 * self.P_bgivena - 1;
        return G;

    def kulczynsky_1 (self):
        K = self.f11 / (self.f10 + self.f01);
        return K;

    def coverage(self):
        return self.P_a;

    def prevalence(self):
        return self.P_b;

    def relative_risk(self):
        RR = self.P_bgivena / self.P_bgivenaprime;
        return RR;

    def piatetsky_shapiro(self):
        return self.P_ab - self.P_a * self.P_b;

    def novelty(self):
        return self.piatetsky_shapiro();

    def yules_q(self):
        YQ = (self.f11 * self.f00 - self.f10 * self.f01) / (self.f11 * self.f00 + self.f10 * self.f01);
        return YQ;

    def yules_y(self):
        YY = (np.sqrt(self.f11 * self.f00) - np.sqrt(self.f10 * self.f01)) / (np.sqrt(self.f11 * self.f00) + np.sqrt(self.f10 * self.f01));
        return YY;

    def cosine(self):
        cosine = self.P_ab / np.sqrt(self.P_a * self.P_b);
        return cosine;

    def least_contradiction(self):
        LC = (self.f11 - self.f10) / (self.f11 + self.f01);
        return LC;

    def odd_multiplier(self):
        OM = (self.P_ab * self.P_bprime) / (self.P_b * self.P_abprime);
        return OM;

    def confirm_descriptive(self):
        CD = self.P_ab - self.P_abprime;
        return CD;

    def confirm_causal(self):
        CC = self.P_ab + self.P_aprimebprime - 2 * self.P_abprime;
        return CC;

    def certainty_factor(self):
        CF = 1 - self.P_abprime / (self.P_a * self.P_bprime)
        return CF;
    
    def loevinger(self):
        return self.certainty_factor();

    def conviction(self):
        conviction = (self.P_a * self.P_bprime) / self.P_abprime;
        return conviction;

    def information_gain(self):
        IG = np.log2(self.P_ab / (self.P_a * self.P_b));
        return IG;

    def laplace_correction(self):
        k = 2;
        LC = (self.f11 + 1) / (self.f11 + self.f10 + k);
        return LC;

    def klosgen(self):
        KL = np.sqrt(self.P_a) * (self.P_bgivena - self.P_b);
        return KL;

    def zhang(self):
        den_1 = self.P_ab * (1 - self.P_b);
        den_2 = self.P_b * (self.P_a - self.P_ab);
        if den_1 > den_2:
            ZH = (self.P_ab - self.P_a * self.P_b)/den_1;
        else:
            ZH = (self.P_ab - self.P_a * self.P_b)/den_2;
        return ZH;

    def normalized_mutual_information(self):
        MI = self.mutual_information();
        NMI = MI / (-self.P_a * np.log2(self.P_a) - self.P_aprime * np.log2(self.P_aprime));
        return NMI;

    def one_way_support(self):
        OWS = self.P_bgivena * np.log2(self.P_bgivena / self.P_b);
        return OWS

    def two_way_support(self):
        TWS = self.P_ab * np.log2(self.P_bgivena / self.P_b);
        return TWS;

    def implication_index(self):
        prod = self.P_a * self.P_bprime; 
        IIN = np.sqrt(self.N) * (self.P_abprime - prod) / np.sqrt(prod);
        return IIN;
    
    def gini_index(self):
        GI = ( self.P_a * ( self.P_bgivena ** 2 + self.P_bprimegivena ** 2 ) +
        self.P_aprime * ( self.P_bgivenaprime ** 2 + self.P_bprimegivenaprime ** 2 ) -
        (self.P_b ** 2) - (self.P_bprime ** 2) );
        return GI
    
    def goodman_kruskal(self):
        P1 = self.P_ab;
        P2 = self.P_abprime;
        P3 = self.P_aprimeb;
        P4 = self.P_aprimebprime;
        numerator = (max(P1, P2) + max(P3, P4) + max(P1, P3) + max(P2, P3)
                     - max(self.P_a, self.P_aprime) - max(self.P_b, self.P_bprime));
        denominator = 2 - max(self.P_a, self.P_aprime) - max(self.P_b, self.P_bprime);
        
        GK = numerator / denominator
        return GK;
    
    def leverage(self):
        L = self.P_bgivena - self.P_a * self.P_b;
        return L;
    
    def kappa(self):
        numerator = (self.P_ab + self.P_aprimebprime 
                     - (self.P_a * self.P_b) 
                     - (self.P_aprime * self.P_bprime));
        denominator = 1 - (self.P_a * self.P_b) - (self.P_aprime * self.P_bprime);

        K = numerator / denominator;
        return K;
    
    def putative_causal_dependency(self):
        PCD = ((0.5 * (self.P_bgivena - self.P_b))
               + (self.P_aprimegivenbprime - self.P_aprime)
               - (self.P_bprimegivena - self.P_bprime)
               - (self.P_agivenbprime - self.P_a));
        return PCD;
    
    def example_counterexample_rate(self):
        ECR = 1 - self.f10 / self.f11;
        return ECR;
    
    def confirmed_confidence_causal(self):
        CCC = 0.5 * (self.P_bgivena + self.P_aprimegivenbprime) - self.P_bprimegivena;
        return CCC;