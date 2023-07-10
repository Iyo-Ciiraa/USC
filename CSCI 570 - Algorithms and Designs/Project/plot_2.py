import matplotlib.pyplot as plt

time_basic = [0,
9.457826614379883,
7.973670959472656,
12.509822845458984,
71.02394104003906,
104.75730895996094,
218.09124946594238,
375.22006034851074,
563.1668567657471,
820.6195831298828,
1695.9967613220215,
2596.202850341797,
3312.2427463531494,
4477.813243865967,
5692.193269729614
]
time_efficient = [0,
2.164602279663086,
8.53419303894043,
31.242847442626953,
66.49065017700195,
115.3421401977539,
268.0926322937012,
439.4388198852539,
695.1818466186523,
1040.4798984527588,
1967.6082134246826,
2793.6649322509766,
4488.720417022705,
5796.859264373779,
6869.813442230225
]
problem_size = [16,
64,
128,
256,
384,
512,
768,
1024,
1280,
1536,
2048,
2560,
3072,
3584,
3968,
]

memory_basic = [13916,
14092,
14256,
14852,
15492,
15748,
17396,
17376,
18204,
18304,
17604,
18476,
18908,
19944,
19148
]

memory_efficient = [14204,
14200,
14084,
14216,
14212,
14208,
14064,
14164,
14244,
14052,
14188,
14160,
14216,
14052,
14180
]
plt.suptitle('Analysis', fontweight = 'bold', fontsize = 10)

plt.subplot(2, 1, 1)
plt.plot(problem_size, time_basic, color = 'green', label = 'Basic')
plt.plot(problem_size, time_efficient, color = 'red', label = 'Efficient')
plt.xlabel('Problem Size', fontsize = 7, fontweight = 'bold')
plt.ylabel('Time', fontsize = 7, fontweight = 'bold')
plt.title('Time vs. Problem Size', fontsize = 8, loc = 'left')
plt.legend(prop = {'size' : 6})

plt.subplot(2, 1, 2)
plt.plot(problem_size, memory_basic, color = 'green', label = 'Basic')
plt.plot(problem_size,memory_efficient, color = 'red', label = 'Efficient')
plt.xlabel('Problem Size', fontsize = 7, fontweight = 'bold')
plt.ylabel('Memory', fontsize = 7, fontweight = 'bold')
plt.title('Memory vs. Problem Size', fontsize = 8, loc = 'left')
plt.legend(prop = {'size' : 6})

plt.subplots_adjust(left=0.15, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4)
plt.show()