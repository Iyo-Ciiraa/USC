import matplotlib.pyplot as plt

time_basic = [1.001119613647461,
1.032114028930664,
6.471395492553711,
21.741867065429688,
64.60261344909668,
83.69588851928711,
195.88685035705566,
359.43078994750977,
525.698184967041,
813.5907649993896,
1440.735101699829,
2268.460512161255,
4124.944686889648,
5429.443597793579,
5781.914472579956
]
time_efficient = [0,
2.164602279663086,
4.054784774780273,
15.633106231689453,
48.05898666381836,
113.4183406829834,
255.50055503845215,
475.6619930267334,
735.7320785522461,
1214.0331268310547,
1967.7352905273438,
3069.2052841186523,
4386.900186538696,
5909.696340560913,
7441.009283065796
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

memory_basic = [13988,
14004,
14228,
14768,
15496,
15572,
17352,
17624,
18348,
18300,
17592,
18340,
18908,
19672,
19244
]

memory_efficient = [14184,
14192,
14028,
14116,
14040,
14060,
14168,
14196,
14176,
14236,
14032,
14024,
14192,
14168,
14200
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