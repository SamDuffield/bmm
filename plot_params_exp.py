import matplotlib.pyplot as plt
import numpy as np
import pickle

sim = False
params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.12647467, 0.13069739, 0.13299723, 0.13402539, 0.13438583,
       0.13398806, 0.13373651, 0.13349616, 0.133643  , 0.13377113,              
       0.13382329, 0.13331461, 0.13346924, 0.1335121 , 0.13360461,              
       0.13312256, 0.13300212, 0.13308532, 0.13319074, 0.13327864,              
       0.13299082, 0.13318676, 0.13285037, 0.13231158, 0.13223606,              
       0.13198401, 0.13178399, 0.13144418, 0.13138585, 0.13170469,              
       0.13148925, 0.13179542, 0.13180117, 0.13218665, 0.13225388,              
       0.13258568, 0.13246797, 0.13213118, 0.13190626, 0.13157687]), 'lambda_speed': np.array([0.1       , 0.08878175, 0.08221098, 0.07951065, 0.0786044 ,
       0.07976388, 0.08084575, 0.08199646, 0.08131367, 0.0808153 ,              
       0.08009263, 0.08130629, 0.08084219, 0.08045435, 0.07983895,              
       0.08075183, 0.08194811, 0.08122353, 0.08076125, 0.08002065,              
       0.08142254, 0.08050642, 0.08128949, 0.0816716 , 0.08231711,              
       0.08253037, 0.0827606 , 0.08275678, 0.08338643, 0.08180134,              
       0.08217859, 0.08236855, 0.08188342, 0.08121958, 0.08103344,              
       0.0803914 , 0.08130038, 0.08173423, 0.0821147 , 0.08237913])}, 'deviation_beta': np.array([0.1       , 0.08092358, 0.0669153 , 0.05973189, 0.05155055,    
       0.05053078, 0.05000102, 0.05567248, 0.0560041 , 0.05607126,              
       0.05013621, 0.05592796, 0.05593496, 0.05614488, 0.05024485,              
       0.04992194, 0.05566986, 0.05573958, 0.05598133, 0.05005928,              
       0.05602814, 0.05007765, 0.04976902, 0.04932198, 0.04948861,              
       0.04909261, 0.04928343, 0.0489397 , 0.0553265 , 0.04950773,              
       0.04923668, 0.05353535, 0.05533361, 0.05428786, 0.05566435,              
       0.05007742, 0.04981148, 0.04957155, 0.04929595, 0.04955819]), 'gps_sd': np.array([7.        , 6.3825907 , 6.04282961, 5.84881813, 5.73590889,
       5.66401117, 5.64934122, 5.65569945, 5.67516552, 5.69273626,              
       5.65956429, 5.66320077, 5.6605009 , 5.65821173, 5.63140864,              
       5.62291219, 5.6385823 , 5.62568927, 5.65367725, 5.61298126,              
       5.66263109, 5.6220483 , 5.61406479, 5.61811794, 5.61169722,              
       5.59284126, 5.59946025, 5.61611878, 5.65537518, 5.62643844,              
       5.62410186, 5.68317898, 5.69740579, 5.67455591, 5.68857257,              
       5.68120703, 5.62070397, 5.60014144, 5.5989139, 5.61316532])}

params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.12647467, 0.13080951, 0.13248298, 0.13342224, 0.1338148 ,
       0.13404028, 0.13412131, 0.1342779 , 0.13435583, 0.13434215,
       0.13442292, 0.13443975, 0.13444228, 0.13450181, 0.1345477 ,
       0.13459571, 0.13452436, 0.13455399, 0.13446409, 0.13441802,
       0.13436105, 0.13433071, 0.1343302 , 0.13431669, 0.1342762 ,
       0.13422895, 0.13418434, 0.13415429, 0.13412625, 0.13408632,
       0.13402644, 0.13394094, 0.13389017, 0.13382768, 0.13378261,
       0.1337267 , 0.13369974, 0.1336741 , 0.13361132, 0.13356769,
       0.13352891, 0.13351107, 0.13348643, 0.13344572, 0.13341329,
       0.13339601, 0.13339796, 0.1333555 , 0.13334259, 0.13330252,
       0.13329451, 0.13326781, 0.13324619, 0.13323961, 0.13322903,
       0.13322713, 0.13323909, 0.13320424, 0.13318898, 0.13313939,
       0.13313357, 0.13310324, 0.13309924, 0.13307824, 0.13304261,
       0.13301312, 0.13300989, 0.13302508, 0.13305021, 0.13305177,
       0.13304339, 0.13302674, 0.13302443, 0.13302313, 0.13304043,
       0.13303282, 0.13302916, 0.13303362, 0.13302222, 0.13300071,
       0.13301404, 0.13301369, 0.13300774, 0.1329943 , 0.13298128,
       0.13297947, 0.13295355, 0.13295596, 0.13298135, 0.13299197,
       0.13299739, 0.13298346, 0.13297425, 0.13296824, 0.13297492,
       0.13297702, 0.13296523, 0.13296281, 0.13295965, 0.13295647,
       0.13293033, 0.13291462, 0.1328889 , 0.13288915, 0.13287994,
       0.1328544 , 0.13284164, 0.1328289 , 0.13282691, 0.13280686,
       0.13280422, 0.13281689, 0.13279359, 0.13279196, 0.13280115,
       0.13279571, 0.13279813, 0.13278632, 0.13276271, 0.13277245,
       0.13277168, 0.1327628 , 0.1327441 , 0.13273994, 0.13274156,
       0.13274375, 0.13273679, 0.13273262]), 'lambda_speed': np.array([0.1       , 0.0887157 , 0.08382818, 0.08149168, 0.08014287,
       0.07936351, 0.07887578, 0.07866533, 0.07849814, 0.07833382,
       0.07828724, 0.07822295, 0.07817721, 0.07819188, 0.0780993 ,
       0.07814211, 0.07816845, 0.07840756, 0.07840102, 0.07855236,
       0.07863794, 0.07871582, 0.078691  , 0.07871965, 0.07876591,
       0.07890515, 0.07892119, 0.07898456, 0.07906701, 0.07915488,
       0.07923154, 0.07936021, 0.07939118, 0.07943271, 0.07946929,
       0.07961577, 0.07966402, 0.07972441, 0.07976587, 0.07982628,
       0.07988344, 0.07996418, 0.08008923, 0.08010066, 0.08018464,
       0.08024556, 0.08024311, 0.08027016, 0.08029962, 0.08028534,
       0.08022079, 0.08021542, 0.08014712, 0.08016089, 0.08011302,
       0.08013859, 0.08009183, 0.08006089, 0.08005584, 0.08003943,
       0.08010487, 0.08013694, 0.08016451, 0.08011627, 0.0800771 ,
       0.08010301, 0.08012549, 0.08015346, 0.08012249, 0.08008999,
       0.08011878, 0.08007695, 0.08010448, 0.08011961, 0.08013546,
       0.0800804 , 0.08004669, 0.08003415, 0.08002038, 0.08004906,
       0.08004652, 0.0800847 , 0.080108  , 0.0800734 , 0.08005064,
       0.0800464 , 0.08007384, 0.08010628, 0.08010523, 0.08009255,
       0.08006835, 0.08009462, 0.08007302, 0.08009816, 0.08012107,
       0.08009631, 0.08007009, 0.08006056, 0.08005854, 0.08006261,
       0.08005541, 0.08006589, 0.08007048, 0.0801354 , 0.0801441 ,
       0.08017598, 0.08020847, 0.08022905, 0.08020801, 0.08022087,
       0.08023518, 0.08021656, 0.08018126, 0.08020742, 0.08019519,
       0.08017887, 0.08017857, 0.08017154, 0.08019374, 0.08018157,
       0.08019206, 0.08023116, 0.0802509 , 0.08026994, 0.08025417,
       0.0802449 , 0.08023422, 0.0802599 ])}, 'deviation_beta': np.array([0.1       , 0.08096656, 0.07025502, 0.06470061, 0.06141903,
       0.05947311, 0.0582848 , 0.05753086, 0.05705696, 0.05673321,
       0.05647489, 0.05628706, 0.05616185, 0.05611711, 0.05447211,
       0.05328134, 0.05238478, 0.05312728, 0.05228675, 0.05296678,
       0.05348238, 0.05387079, 0.05292315, 0.05214452, 0.05154442,
       0.05224109, 0.05161357, 0.05115654, 0.05079783, 0.05050258,
       0.0502253 , 0.0510688 , 0.05068272, 0.05039967, 0.04972243,
       0.05056852, 0.0502932 , 0.05007291, 0.04989426, 0.0498167 ,
       0.04959621, 0.04950492, 0.05028817, 0.05001869, 0.0507017 ,
       0.05127581, 0.05095848, 0.05152047, 0.05202465, 0.05243525,
       0.05195986, 0.05234372, 0.05184329, 0.05223172, 0.05177302,
       0.05216002, 0.05141828, 0.05106161, 0.05076434, 0.05055761,
       0.051053  , 0.05156833, 0.05195942, 0.05155425, 0.05122542,
       0.05167636, 0.05202754, 0.05241493, 0.05199152, 0.05159401,
       0.05196755, 0.05158968, 0.05196473, 0.0522912 , 0.05258034,
       0.05216503, 0.05178692, 0.0514426 , 0.05113993, 0.05154607,
       0.051254  , 0.05163489, 0.05197208, 0.0516181 , 0.05130863,
       0.05105291, 0.05143703, 0.05178922, 0.05146426, 0.05106234,
       0.05130339, 0.05168324, 0.0513908 , 0.05172875, 0.05204089,
       0.05171749, 0.05140052, 0.0511226 , 0.05090015, 0.05068553,
       0.05047979, 0.05030921, 0.05015795, 0.05062918, 0.05045046,
       0.05083257, 0.05119299, 0.05152926, 0.05125657, 0.05158342,
       0.05189841, 0.05157806, 0.05131315, 0.05163488, 0.05136311,
       0.0511104 , 0.05089893, 0.0506852 , 0.05107037, 0.05061498,
       0.05047143, 0.05085581, 0.05122302, 0.05154343, 0.05128753,
       0.05107678, 0.05088725, 0.0512361 ]), 'gps_sd': np.array([7.        , 6.35586038, 5.99002221, 5.82667639, 5.71806016,
       5.69359831, 5.66807513, 5.66987343, 5.68351162, 5.67220325,
       5.67335026, 5.67361658, 5.64423348, 5.64676409, 5.6155078 ,
       5.59748635, 5.59332834, 5.62365268, 5.6305887 , 5.6429993 ,
       5.64945893, 5.66594872, 5.62236613, 5.61178715, 5.59235387,
       5.61514262, 5.60286618, 5.62023885, 5.61829269, 5.59435515,
       5.56554725, 5.62926766, 5.60266104, 5.61442377, 5.61299667,
       5.59932165, 5.59977527, 5.58242554, 5.60599763, 5.61627982,
       5.61525944, 5.6233634 , 5.62811724, 5.59316929, 5.64276369,
       5.65223828, 5.63041724, 5.64287284, 5.66219073, 5.64133187,
       5.60755256, 5.60834615, 5.60032279, 5.65094159, 5.62474763,
       5.65692301, 5.61249705, 5.58312972, 5.5777091 , 5.58995321,
       5.625202  , 5.65888486, 5.6584318 , 5.63265488, 5.60965911,
       5.62937971, 5.63369668, 5.65888887, 5.61832274, 5.60215538,
       5.64289787, 5.61084807, 5.6545458 , 5.64533689, 5.64011142,
       5.60833735, 5.59696752, 5.60125369, 5.60498284, 5.63555977,
       5.62484836, 5.65343285, 5.64730137, 5.59688627, 5.60596718,
       5.58171781, 5.62003283, 5.65860807, 5.63317   , 5.61212634,
       5.650767  , 5.63653496, 5.62024975, 5.6404854 , 5.64430017,
       5.60631459, 5.58712037, 5.60155296, 5.58664357, 5.60412363,
       5.57873323, 5.58273671, 5.61286325, 5.64047934, 5.60770647,
       5.61958957, 5.63789922, 5.64767024, 5.63849196, 5.65420921,
       5.66557266, 5.61758651, 5.60203674, 5.62278994, 5.59884825,
       5.59863866, 5.6047541 , 5.58819296, 5.59685913, 5.59564458,
       5.61544634, 5.65364329, 5.64879017, 5.63524806, 5.61983586,
       5.61842316, 5.60333201, 5.62951668])}

params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.12647467, 0.1302187 , 0.1316902 , 0.13250631, 0.13288179,
       0.1331786 , 0.13331473, 0.13331742, 0.13332921, 0.13328883,              
       0.13325121, 0.13323987, 0.13320658, 0.13325906, 0.13318675,              
       0.13320123, 0.13317528, 0.1331499 , 0.1330896 , 0.13309751,              
       0.1330779 , 0.13304772, 0.13306908, 0.13303479, 0.13299688,              
       0.13296513, 0.13295103, 0.13294793, 0.13290493, 0.13290643,              
       0.13288211, 0.132889  , 0.1328991 , 0.13288106, 0.13289663,              
       0.13290574, 0.13291126, 0.13292333, 0.13291589, 0.13291836,              
       0.13289668, 0.13290776, 0.13290675, 0.13293656, 0.13292279,              
       0.13291154, 0.1328782 , 0.13289328, 0.13289355, 0.13288054,              
       0.13283396, 0.13284915]), 'lambda_speed': np.array([0.1       , 0.08722124, 0.08097775, 0.07728032, 0.07476691,
       0.07300366, 0.07176919, 0.0707925 , 0.07008513, 0.06956312,              
       0.06913552, 0.06880848, 0.06852505, 0.0683537 , 0.06809766,              
       0.06793688, 0.06776029, 0.06760673, 0.06743353, 0.06731218,              
       0.06721972, 0.06711032, 0.06705029, 0.06695093, 0.06684553,              
       0.06675722, 0.06668098, 0.06661697, 0.06653446, 0.06648586,              
       0.06641826, 0.06637999, 0.06634831, 0.06628735, 0.06626487,              
       0.06623667, 0.06620932, 0.06618261, 0.06614699, 0.06611694,              
       0.06606937, 0.06605543, 0.06600094, 0.065901  , 0.06588582,              
       0.06587563, 0.06584247, 0.06585332, 0.06584364, 0.06581914,              
       0.06576597, 0.06574831])}, 'deviation_beta': np.array([0.1       , 0.08230195, 0.07292084, 0.06741117, 0.06390116,
       0.06058987, 0.0585769 , 0.05709128, 0.05576006, 0.05534696,              
       0.05496393, 0.05470529, 0.05460087, 0.05451454, 0.05451837,              
       0.05450726, 0.05446643, 0.05447642, 0.05447656, 0.05420764,              
       0.05433599, 0.05438318, 0.05442028, 0.05448203, 0.05454398,              
       0.05458818, 0.05459578, 0.05458707, 0.05461788, 0.0546342 ,              
       0.05464525, 0.05466238, 0.05471461, 0.05466916, 0.05468032,              
       0.05470948, 0.05470847, 0.05475566, 0.0547894 , 0.05479351,              
       0.05479736, 0.05480912, 0.05441397, 0.05429836, 0.05433888,              
       0.05441888, 0.05447871, 0.05452983, 0.05460846, 0.05463456,              
       0.05463512, 0.05432444]), 'gps_sd': np.array([7.        , 6.23939837, 5.79668224, 5.60451024, 5.49703642,
       5.37060212, 5.2901081 , 5.25522071, 5.25652103, 5.21176793,              
       5.19410699, 5.20472661, 5.21752874, 5.22623689, 5.22160032,              
       5.21514485, 5.19453113, 5.23486423, 5.21303912, 5.17749057,              
       5.20521628, 5.18294557, 5.18296129, 5.20555367, 5.2196821 ,              
       5.22535782, 5.19740003, 5.18520502, 5.23292222, 5.22139257,              
       5.22914131, 5.21871785, 5.23380972, 5.24726502, 5.22783789,              
       5.23966028, 5.23204297, 5.20384672, 5.18197805, 5.15987618,              
       5.16445753, 5.17624163, 5.17489822, 5.23265403, 5.20422805,              
       5.2183021 , 5.23583517, 5.18678852, 5.21030208, 5.20357274,              
       5.22303915, 5.19678604])}

# Simulated data
# sim = True
# params = pickle.load(open('/Users/samddd/Main/bayesian-map-matching/simulations/cambridge/tuned_sim_params.pickle', 'rb'))
# params = pickle.load(open('/Users/samddd/Main/bayesian-map-matching/simulations/cambridge/tuned_sim_params._0.1_0.05_0.05_3_prcap500_final.pickle', 'rb'))
#
# params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.12647467, 0.13352728, 0.13563493, 0.1363461 , 0.13680604,
#        0.13706412, 0.13734336, 0.1376507 , 0.13790093, 0.13822715,
#        0.13847304, 0.13865541, 0.13883829, 0.13902936, 0.13922137,
#        0.13937613, 0.13959797, 0.13978853, 0.13987911, 0.14002672,
#        0.14010243, 0.14021578, 0.1403004 , 0.14036982, 0.14044573,
#        0.14054459, 0.1405827 , 0.14065552, 0.1407209 , 0.14081909,
#        0.14087992, 0.14094747, 0.14100441, 0.1410549 , 0.14112139,
#        0.14115597, 0.14119469, 0.14129322, 0.14129424, 0.14132965,
#        0.14135054, 0.14140988, 0.14148727, 0.14154381, 0.14159415,
#        0.14165561, 0.14171985]), 'lambda_speed': np.array([0.1       , 0.0739436 , 0.06336797, 0.05815913, 0.05552143,
#        0.05393546, 0.05291699, 0.05226735, 0.05174492, 0.0514462 ,
#        0.0511343 , 0.05082605, 0.05057932, 0.0503856 , 0.05024768,
#        0.05009778, 0.05004654, 0.04999487, 0.04984553, 0.04977708,
#        0.04964277, 0.04957647, 0.04948316, 0.04937956, 0.04929793,
#        0.0492627 , 0.0491617 , 0.04911753, 0.04907581, 0.04906599,
#        0.04902282, 0.04899825, 0.04896457, 0.04892781, 0.04891075,
#        0.04886173, 0.04881766, 0.04885596, 0.04877713, 0.04874515,
#        0.04869682, 0.04869665, 0.04872931, 0.04873236, 0.04873196,
#        0.04874611, 0.04876179])}, 'deviation_beta': np.array([0.1       , 0.07178123, 0.05778047, 0.05125079, 0.04863266,
#        0.04785774, 0.04757534, 0.04758136, 0.04756025, 0.04767784,
#        0.04777918, 0.04783405, 0.0479456 , 0.04795298, 0.0480369 ,
#        0.04807616, 0.04808304, 0.04813162, 0.04821071, 0.04824485,
#        0.04828124, 0.04830583, 0.04832075, 0.04833678, 0.04836524,
#        0.04836811, 0.04838825, 0.04843482, 0.04846269, 0.04844384,
#        0.04843336, 0.0484814 , 0.04853264, 0.04856322, 0.04859   ,
#        0.04860479, 0.04860215, 0.04862118, 0.04863223, 0.04867907,
#        0.04866147, 0.04866756, 0.04866849, 0.04868308, 0.04870459,
#        0.04873472, 0.04872889]), 'gps_sd': np.array([7.        , 5.24028681, 4.17109222, 3.56563876, 3.26107578,
#        3.11322238, 3.03905867, 3.02527181, 3.00064226, 2.99670348,
#        2.99571855, 2.99223675, 2.99335395, 2.99659232, 3.00336285,
#        2.99956131, 2.99161086, 2.990963  , 2.99212789, 2.98595214,
#        3.00086809, 2.99347553, 2.99137939, 2.99161227, 3.00269222,
#        3.00098867, 3.00717933, 3.01076062, 2.98899627, 2.99975137,
#        2.98912355, 3.00038691, 2.99231244, 3.00028342, 2.99401049,
#        3.00070735, 2.99278517, 2.99431949, 2.99315177, 3.00771665,
#        3.00088354, 2.99678013, 2.98577468, 2.99245836, 2.99419961,
#        2.99700142, 2.98893916])}

n_iter = len(params['gps_sd'])

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 10))

axes[0].plot(np.arange(n_iter), np.exp(-15 * params['distance_params']['zero_dist_prob_neg_exponent']))
axes[1].plot(np.arange(n_iter), params['distance_params']['lambda_speed'])
axes[2].plot(np.arange(n_iter), params['deviation_beta'])
axes[3].plot(np.arange(n_iter), params['gps_sd'])

axes[0].set_ylabel(r'$p^0$')
axes[1].set_ylabel(r'$\lambda$')
axes[2].set_ylabel(r'$\beta$')
axes[3].set_ylabel(r'$\sigma_{GPS}$')

if sim:
       line_colour = 'purple'
       axes[0].hlines(0.10, 0, n_iter, colors=line_colour)
       axes[1].hlines(1/20, 0, n_iter,  colors=line_colour)
       axes[2].hlines(0.05, 0, n_iter, colors=line_colour)
       axes[3].hlines(3.0, 0, n_iter, colors=line_colour)

plt.tight_layout()
plt.show()
