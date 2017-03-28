import numpy as np;
from math import erf;
from Measures import map_measures_to_indices;

(measures_dict, measures_arr) = map_measures_to_indices();

def generate_contingency_tables(vals):
    tables = np.zeros(shape=(vals.size**4,4));
    i = 0;
    for tp in vals:
        for tn in vals:
            for fp in vals:
                for fn in vals:
                    tables[i] = [tp,tn,fp,fn];
                    i += 1;
    return tables;


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
        self.scores[self.measures['recall']] = self.recall();
        self.scores[self.measures['precision']] = self.precision();
        self.scores[self.measures['confidence']] = self.confidence();
        self.scores[self.measures['mutual_information']] = self.mutual_information();
        self.scores[self.measures['jaccard']] = self.jaccard();
        self.scores[self.measures['f_measure']] = self.f_measure();
        self.scores[self.measures['odds_ratio']] = self.odds_ratio();
        self.scores[self.measures['specificity']] = self.specificity();
        self.scores[self.measures['negative_reliability']] = self.negative_reliability();
        self.scores[self.measures['sebag_schoenauer']] = self.sebag_schoenauer();
        self.scores[self.measures['accuracy']] = self.accuracy();
        self.scores[self.measures['support']] = self.support();
        self.scores[self.measures['confidence_causal']] = self.confidence_causal();
        self.scores[self.measures['lift']] = self.lift();
        self.scores[self.measures['ganascia']] = self.ganascia();
        self.scores[self.measures['kulczynsky_1']] = self.kulczynsky_1();
        self.scores[self.measures['coverage']] = self.coverage();
        self.scores[self.measures['prevalence']] = self.prevalence();
        self.scores[self.measures['relative_risk']] = self.relative_risk();
        self.scores[self.measures['piatetsky_shapiro']] = self.piatetsky_shapiro();
        self.scores[self.measures['novelty']] = self.novelty();
        self.scores[self.measures['yules_q']] = self.yules_q();
        self.scores[self.measures['yules_y']] = self.yules_y();
        self.scores[self.measures['cosine']] = self.cosine();
        self.scores[self.measures['least_contradiction']] = self.least_contradiction();
        self.scores[self.measures['odd_multiplier']] = self.odd_multiplier();
        self.scores[self.measures['confirm_descriptive']] = self.confirm_descriptive();
        self.scores[self.measures['confirm_causal']] = self.confirm_causal();
        self.scores[self.measures['certainty_factor']] = self.certainty_factor();
        self.scores[self.measures['loevinger']] = self.loevinger();
        self.scores[self.measures['conviction']] = self.conviction();
        self.scores[self.measures['information_gain']] = self.information_gain();
        self.scores[self.measures['laplace_correction']] = self.laplace_correction();
        self.scores[self.measures['klosgen']] = self.klosgen();
        self.scores[self.measures['zhang']] = self.zhang();
        self.scores[self.measures['normalized_mutual_information']] = self.normalized_mutual_information();
        self.scores[self.measures['one_way_support']] = self.one_way_support();
        self.scores[self.measures['two_way_support']] = self.two_way_support();
        self.scores[self.measures['implication_index']] = self.implication_index();
        self.scores[self.measures['gini_index']] = self.gini_index();
        self.scores[self.measures['goodman_kruskal']] = self.goodman_kruskal();
        self.scores[self.measures['leverage']] = self.leverage();
        self.scores[self.measures['kappa']] = self.kappa();
        self.scores[self.measures['putative_causal_dependency']] = self.putative_causal_dependency();
        self.scores[self.measures['example_counterexample_rate']] = self.example_counterexample_rate();
        self.scores[self.measures['confirmed_confidence_causal']] = self.confirmed_confidence_causal();
        self.scores[self.measures['added_value']] = self.added_value();
        self.scores[self.measures['collective_strength']] = self.collective_strength();
        self.scores[self.measures['j_measure']] = self.j_measure();
        self.scores[self.measures['dependency']] = self.dependency();
        
        # self.scores = np.round(self.scores, decimals = 6);


        # self.scores[self.measures['theil_uncertainty_coefficient']] = self.theil_uncertainty_coefficient();
        # self.scores[self.measures['TIC']] = self.TIC();
        # self.scores[self.measures['logical_necessity']] = self.logical_necessity();
        # self.scores[self.measures['kulczynsky_2']] = self.kulczynsky_2();
        # self.scores[self.measures['k_measure']] = self.k_measure();
        # self.scores[self.measures['interestingness_weighting_dependency']] = self.interestingness_weighting_dependency();
        # self.scores[self.measures['directed_information_ratio']] = self.directed_information_ratio();
        # self.scores[self.measures['chi_square']] = self.chi_square();
        # self.scores[self.measures['dilated_chi_square']] = self.dilated_chi_square();
        # self.scores[self.measures['conditional_entropy']] = self.conditional_entropy();
        # self.scores[self.measures['complement_class_support']] = self.complement_class_support();
        # self.scores[self.measures['intensity_of_implication']] = self.intensity_of_implication();
        # self.scores[self.measures['correlation_coefficient']] = self.correlation_coefficient();
        
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

    def added_value(self):
        AV = self.P_bgivena - self.P_b;
        return AV;

    def collective_strength(self):
        good_events = self.P_ab + self.P_aprimebprime;
        E_good_events = (self.P_a * self.P_b) + (self.P_aprime * self.P_bprime);
        CS = (good_events * (1 - E_good_events)) / (E_good_events * (1 - good_events));
        return CS;
    
    def j_measure(self):
        JM = (self.P_ab * np.log2(self.P_bgivena / self.P_b) 
                + self.P_abprime * np.log2(self.P_bprimegivena / self.P_bprime));
        return JM;

    def dependency(self):
        return self.added_value();

    def theil_uncertainty_coefficient(self):
        MI = self.mutual_information();
        TUC = MI / ( -1 * self.P_b * np.log2(self.P_b)
                     - self.P_bprime * np.log2(self.P_bprime) );
        return TUC;

    def TIC(self):
        factor1 = self.directed_information_ratio();
        table2 = contingency_table([self.f01, self.f00, self.f11, self.f10]);
        factor2 = table2.directed_information_ratio();
        # print(factor1,factor2);
        T = np.sqrt(factor1 * factor2);
        return T;

    def logical_necessity(self):
        LN = (1 - self.P_agivenb) / (1 - self.P_agivenbprime);
        return LN;

    def kulczynsky_2(self):
        K2 = ( self.P_ab / self.P_a + self.P_ab / self.P_b) / 2;
        return K2;

    def k_measure(self):
        KM = ( self.P_bgivena * np.log2(self.P_bgivena / self.P_b) +
                self.P_bprimegivenaprime * np.log2(self.P_bprimegivenaprime / self.P_bprime) -
                self.P_bgivena * np.log2(self.P_bgivena / self.P_bprime) - 
                self.P_bprimegivenaprime * np.log2(self.P_bprimegivenaprime / self.P_b) );
        return KM;

    def interestingness_weighting_dependency(self):
        l = 2;
        m = 1;
        IWD = ( ( self.P_ab / ( self.P_a * self.P_b ) )**l - 1 ) * self.P_ab**m;
        return IWD;

    def directed_information_ratio(self):
        if (self.P_b == 1):
            DIR = -np.inf;
        elif ((self.P_b <= 0.5) & (self.P_bgivena <= 0.5)):
            DIR = 0;
        elif ((self.P_b <= 0.5) & (self.P_bgivena > 0.5)):
            DIR = 1 + self.P_bgivena * np.log2(self.P_bgivena) + self.P_bprimegivena * np.log2(self.P_bprimegivena);
        elif ((self.P_b > 0.5) & (self.P_bgivena <= 0.5)):
            DIR = 1 + 1 / (self.P_b * np.log2(self.P_b) + self.P_bprime * np.log2(self.P_bprime));
        elif ((self.P_b > 0.5) & (self.P_bgivena > 0.5)):
            DIR = 1 - ( self.P_bgivena * np.log2(self.P_bgivena) + self.P_bprimegivena * np.log2(self.P_bprimegivena) ) / ( self.P_b * np.log2(self.P_b) + self.P_bprime * np.log2(self.P_bprime) );
        else:
            raise ValueError('cannot compute DIR');            
        return DIR;

    def chi_square(self):
        numerator = self.N * ( self.P_ab - self.P_a * self.P_b )**2;
        denominator = self.P_a * self.P_aprime * self.P_b * self.P_bprime;
        CS = numerator/denominator;
        return CS;

    def dilated_chi_square(self):
        alpha = 1;
        numerator = self.P_a * self.P_aprime * self.P_b * self.P_bprime;
        op1 = min(self.P_a, self.P_aprime);
        op2 = min(self.P_b, self.P_bprime);
        op3 = max(self.P_a, self.P_aprime);
        op4 = max(self.P_b, self.P_bprime);
        denominator = ( min(op1, op2) * min(op3, op4) )**2;
        DCS = (numerator / denominator)**alpha * self.chi_square();
        return DCS;

    def conditional_entropy(self):
        CE = -1 * self.P_bgivena * np.log2(self.P_bgivena) - self.P_bprimegivena * np.log2(self.P_bprimegivena);
        return CE;

    def complement_class_support(self):
        CCS = self.P_abprime * self.P_bprime;
        return CCS;

    def intensity_of_implication(self):
        IIN_factor = self.implication_index() / np.sqrt(2);
        # IIN_factor_s = IIN_factor**2;
        # p = 1 - np.power ( np.exp, ( -1 * IIN_factor_s * ( 4/np.pi + 0.147 * IIN_factor_s) / (1 + 0.147 * IIN_factor_s) ) );
        # IIM = 0.5 - 0.5 * np.sign(IIN_factor)
        IIM = 0.5 - 0.5 * erf(IIN_factor);
        return IIM;

    def correlation_coefficient(self):
        numerator = self.P_ab - self.P_a * self.P_b;
        denominator = np.sqrt( self.P_a * self.P_b * self.P_aprime * self.P_bprime );
        CCO = numerator / denominator;
        return CCO;
