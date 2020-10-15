import sklearn.metrics
import numpy 
import matplotlib.pyplot 
# MCC-F1 Funtion:
def mcc_f1(ground_truths, predicted_values):
    
    """
    The function mcc_f1 calculates MCC and F1 scores based on varying thresholds identifed by the PRC function
    based on inputs -  ground truth vaules and predicted values.
    
    Parameters
    ----------
    ground_truths, predicted_values : numpy.ndarray;
                        Input arrays; groundtruth values (True values) and predicted values.
    
    Returns
    -------
    mcc, f1, thresholds: numpy.ndarray;
                           Output arrays; Unit normalized MCC scores and F1 score values 
                           for every threshold.           
    """
    
    mcc = []   
    f1 = []
    
# Generating thresholds with the PRC function
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
                                    ground_truths, predicted_values)

# For every value of the thresholds (cutoffs), calculate MCC and F1 scores :
    for thresh in thresholds:
        y_pred_thresh = predicted_values > thresh  # Positive classes are determined for every predicted value that exceeds the threshold. 
    
        mcc.append((sklearn.metrics.matthews_corrcoef(
                    ground_truths, y_pred_thresh)+1)*0.5) # get normalised MCC: change the range of MCC from [-1, 1] to [0, 1] 
        
        f1.append(sklearn.metrics.f1_score(
                    ground_truths, y_pred_thresh))
        
    mcc = numpy.array(mcc)
    f1 = numpy.array(f1)  
    
  # Resulting values are an array of MCC & F1 scores across different thresholds (cutoffs)  
    return mcc,f1, thresholds

# MCC-F1 Metric:
def MCC_F1_Metric(mcc,f1,thresholds, W=100):
    
    """
    MCC_F1_Metric function computes the MCC-F1 metric i.e. the average distances between the predcition scores to the point of Perfect Performace (1,1) 
    and also results in the best threshold value
    
    Parameters
    ----------
    
    mcc,f1, thresholds : numpy.ndarray;
                            MCC and F1 scores across varying thresholds
    
    W : int, optional;
        Number of subranges- default = 100; 
        larger values of 𝑊 will cause the MCC-𝐹1 metric to capture the performance of a classifier more accurately.
    
    Returns
    -------
    MCC_F1_Met : float;
                The ratio of the average distance of the MCC-F1 score to the Point of Perfect Performace (1,1)
                A metric to compare classifier performance.
    
    T_index : int;
                Index of the best prediction score threshold (𝑇), 
                Index of the point on the MCC-𝐹1 curve closest to the point of perfect performance (1,1).
    
    """
    
     
    mcc_intervals = numpy.linspace(numpy.min(mcc), numpy.max(mcc),W) # breaking into subranges
    
    # Computing MCC values per subrange
    
    n = numpy.zeros_like(mcc_intervals) # 'n' is the number of points per subrange
    for i in range(W-1):
        for j in mcc:
            if j >= mcc_intervals[i] and j < mcc_intervals[i+1]:
                n[i] = n[i] + 1

    # Calculating the distances between points in a subrange to the Point of perfect performance (1,1)            
    # Di = numpy.zeros_like(mcc)
    Di = numpy.sqrt(((mcc-1)**2)+((f1-1)**2)) # calculating the Euclidean distance 𝐷𝑖            
    sum_Di = numpy.zeros_like(mcc_intervals)
    index = -1
    for value in mcc:
        index += 1
        for i in range(W-1):
                 if value >= mcc_intervals[i] and value < mcc_intervals[i+1]:
                        sum_Di[i] = sum_Di[i] + Di[index]

# Mean Distance across subranges
    mean_Di = numpy.array(sum_Di/n)
    P = 0  
    mean_Di_sum = 0
    for i in mean_Di:
        if not numpy.isnan(i):
            P += 1 
            mean_Di_sum += i # addition of all the means across subranges that have atleast 1 MCC value.

    grand_avg = mean_Di_sum/P # P = total number of subranges that have atleast 1 MCC value
   
    """ 
    Compare the grand average distance to √2 (The distance between the point of worst performance (0,0) and 
    the point of perfect performance (1,1) is √2).That is the maximum possible distance between a point on the MCC-𝐹1 curve
    The ratio between the grand avgerage distance and √2 is taken.
    This ratio ranges between 0 and 1 (worst value = 0; best value = 1). To get the MCC-𝐹1 score, we subtract this ratio from 1
    """

    MCC_F1_Met = 1 - (grand_avg /numpy.sqrt(2))
  
    
    # Finding the best threshold 𝑇 the point on the MCC-𝐹1 curve closest to the point of perfect performance (1,1).
   
    result = numpy.where(Di == Di.min())
    T_index = result[0][0]
    
    print("MCC-F1 metric = {0:.2f}".format(MCC_F1_Met), "Best Threshold = {0:.2f}".format(thresholds[T_index]))
    
    return MCC_F1_Met,T_index

# Generating a plot (the overall canvas);
axis = None
def axis_plotting(total_plots): 
    global axis
    if axis == None:
        fig = matplotlib.pyplot.figure(figsize=(7,7))
        axis = []        
        for i in range(1,total_plots+1):
            ax = fig.add_subplot(total_plots, 1, i)
            axis.append(ax)
            
    
def plotting(mcc, f1, thresholds,MCC_F1_Met,T_index,classifer_name = "",total_plots = 1,ax = 0):
    
        """
        Plotting function - fuction to plot the MCC and F1 scores (across varying Thresholds), and highlight the the MCC_F1 metric with the best threshold
        For multiple plots, 'total_plots' should be > 1 AND 'ax' (< total_plots) starting from  1
    
        Parameters
        ----------
        mcc,f1 : numpy.ndarray;
            MCC and F1 scores across varying thresholds
    
        MCC_F1_Met : float;
                The ratio of the average distance of the MCC-F1 score to the Point of Perfect Performace (1,1)
                A metric to compare classifier performance.
    
        T_index : int;
                Index of the best prediction score threshold (𝑇), 
                Index of the point on the MCC-𝐹1 curve closest to the point of perfect performance (1,1).
                
        classifer_name : str, optional
                    default = ""
    
        total_plots: int, optional
                    Number of sub-plots to generate; default = 1
    
        ax : int, optional
            Index of subplot ([1,2,3...]); default = None
    
                            
        Returns
        -------
        
        MCC-F1 curve: 
            for multiple sublplots, adjust total_plots and ax input value.
                        
        """
        
        global axis
        if total_plots > 1 and ax < total_plots:
            axis_plotting(total_plots) #calling fucntion to
            if ax == 0:
                subplot = matplotlib.pyplot.gca()
            else:
                subplot = axis[ax-1]
        elif ax > total_plots:
            print("Index inconsistent with total_plots")
        else:
            axis = None
            subplot = matplotlib.pyplot.gca()
        
        
        
      
        # Labelling the plot;
        subplot.plot(f1,mcc)
        subplot.annotate(classifer_name + "_MCC_Met - {0:.2f}".format(
            MCC_F1_Met), 
                 (f1[T_index],mcc[T_index]), 
                 textcoords="offset points", 
                 xytext=(0,10),
                 ha='left') 
        
        subplot.scatter(f1[T_index],mcc[T_index],color='red')
        subplot.set(xlabel='F1 score',
               ylabel='unit−normalized MCC',title='MCC-F1 curve')
        
        # Inserting Random line
        subplot.axhline(0.5,linewidth=0.7, color='green',linestyle='--')
        subplot.annotate('Random line',xy =(0,0.5),xytext=(250,-15),
                    color='g', xycoords = subplot.get_yaxis_transform(), 
                    textcoords="offset points", va="bottom")


        # Displaying and labelling points of performances.

        subplot.scatter(1,1,color = 'black');subplot.scatter(0,0,color = 'black')
        subplot.annotate('Point of Perfect performance',(1,1),
                    textcoords="offset points",xytext=(-9,-2),ha='right')
        subplot.annotate('Point of Worst performance',(0,0)
                    ,textcoords="offset points",xytext=(9,2),ha='left')
        
        subplot.set_xticks(numpy.arange(0, 1.1, 0.25)); subplot.set_yticks(numpy.arange(0, 1.1, 0.25))
        subplot.set_xlim(-0.05,1.05); subplot.set_ylim(-0.05,1.05)
        subplot.grid(alpha = 0.3)