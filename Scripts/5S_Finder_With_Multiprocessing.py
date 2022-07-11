from scipy.stats import norm as nr
import matplotlib.pyplot as plt
import scipy.fft as spt
import numpy as np
from tqdm import tqdm
import scipy.optimize as so
import multiprocessing
from multiprocessing import Process, Manager
import datetime
import os


def invF(dt, A, xlimf='none', xlimt='none', plot='yes', output='no', absolute='yes', norm='no',
         positiveoutput='no'):  # same as last cell but for inverse transform
    n = np.size(A)
    f = spt.fftfreq(np.size(A), d=dt)
    t = np.arange(0, np.size(A) * dt / 2, dt)
    # print(f)
    # print(t)
    # print(A)
    if absolute == 'no':
        if norm == 'yes':
            Y0 = (spt.ifft(A)) / np.size(f)
        if norm == 'no':
            Y0 = (spt.ifft(A))
    if absolute == 'yes':
        if norm == 'yes':
            Y0 = abs(spt.ifft(A)) / np.size(f)
        if norm == 'no':
            Y0 = abs(spt.ifft(A))
    Y = []
    for i in range(
            len(Y0) // 2):  # only plots the positive part of the time domain since negative part does not have physical meaning
        Y.append(Y0[i])

    if plot == 'yes':
        plt.subplot(1, 2, 1)
        plt.plot(f, A, '.')
        if xlimf != 'none':
            plt.xlim([-xlimf, xlimf])
        plt.xlabel('Frequency(hz)')
        plt.ylabel('Power')
        plt.title('Frequency Domain')

        plt.subplot(1, 2, 2)
        plt.plot(t, Y, '.')
        if xlimt != 'none':
            plt.xlim([0, xlimt])
        plt.xlabel('Time(s)')
        plt.ylabel('Magnitude')
        plt.title('Time Domain')

    if output == 'yes':
        if positiveoutput == 'yes':
            return Y
        else:
            return Y0  # negative part of time domain is necessary for further transformation


def gendata(T, dt, noise=2, plot=True, output=True):
    n = 4 * int(T // dt)
    F = spt.fftfreq(n + 4, d=dt)
    w = 2 * np.pi * F
    S = (1 / w) ** noise
    S[0] = S[1]
    re = []
    im = []
    for i in range(len(F)):
        if F[i] >= 0:
            r = np.random.normal(0, 1, 2)
            re.append((0.5 * S[i]) ** 0.5 * r[0])
    for i in range(len(re)):
        im.append(re[-i])
    y = np.concatenate([re, im])
    Mi = invF(dt, re, output='yes', absolute='no', positiveoutput='yes', plot='no')
    Ma = np.array(Mi)
    M = np.real(Ma)/np.amax(abs(np.real(Ma)))
    t = np.arange(0, np.size(M) * dt, dt)

    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
        ax1.plot(F, y, '.')
        ax1.set_title('Linear')
        ax1.set_xlabel('Frequency(hz)')
        ax1.set_ylabel('sqrt Power')
        ax2.loglog(F, y, '.')
        ax2.set_xlabel('Frequency(hz)')
        ax2.set_title('Log-log')
        fig.suptitle('Sample Light Curve Frequency Space', fontsize=16)

        plt.figure()
        plt.plot(t, M, '.')
        plt.xlabel('Time(s)')
        plt.ylabel('Magnitude')
        plt.title('Sample Light Curve Time Space', size=16)

    if output == True:
        tr = []
        mr = []
        for i in range(len(t)):
            tr.append(t[i])
            mr.append(np.real(M[i]))
        return tr, mr


def getcorrelation(datafile, Detector, N, a, b, dt, strength=1, plot1=False, plot2=False, showrs=False, printoutput=False, histogram=False, crit=False, r=[], error=[] ,ostrengths=[], grms=[] ,foldername=''):
    data = np.genfromtxt(datafile, delimiter=',')
    tdata = []
    mdata = []
    for i in data:
        tdata.append(i[0])
        mdata.append(abs(i[Detector]))
    T = b - a
    ttdata0 = []
    mmdata = []
    for i in range(len(tdata)):
        if tdata[i] > a and tdata[i] <= b:
            ttdata0.append(tdata[i])
            mmdata.append(mdata[i])
    ttdata = []
    for i in range(len(ttdata0)):
        ttdata.append(ttdata0[i] - ttdata0[0])
    norm = np.array(mmdata) - (np.sum(mmdata) / np.size(mmdata))
    rms = (np.sum(np.array(norm) ** 2) / np.size(norm)) ** 0.5

    curves = []
    for i in tqdm(range(N)):
        if plot1 == True:
            #plt.figure()
            curve = gendata(T, dt, plot=True)
        else:
            curve = gendata(T, dt, plot=False)
        curves.append(curve[1])
    injecteds = []
    for i in curves:
        yinjected = []
        for j in range(len(mmdata)):
            yinjected.append(abs(mmdata[j] + strength * rms * i[j]))
        if plot2 == True:
            plt.figure()
            plt.plot(ttdata, yinjected, '.')
            plt.title('Trimmed ECM Data with Injected Sample Light Curve')
            plt.xlabel('Time(s)')
            plt.ylabel('Average Absolute Magnitude(V)')
        injecteds.append(yinjected)
    rs = []
    for i in range(len(injecteds)):
        corr_matrix = np.corrcoef(curves[i], injecteds[i])
        corr = corr_matrix[0, 1]
        rs.append(corr)
    mu = np.mean(rs)
    sigma = np.std(rs, ddof=0)

    def gaussian(x, sigma, mu):
        p = (1 / (sigma * (2 * np.pi) ** 0.5)) * np.e ** (-1 * ((x - mu) ** 2) / (2 * sigma ** 2))
        return p

    (gmu, gsigma) = nr.fit(rs)
    Rrange = np.linspace(0, 1, 1000)
    yR = []
    for i in Rrange:
        yR.append(gaussian(i, gsigma, gmu))

    if showrs == True:
        print(rs)
    rmean = np.sum(rs) / N
    if printoutput == True:
        print(rmean)
    if histogram == True:
        plt.figure()
        t = plt.hist(rs, 10, )
        if crit==True:
            plt.title('Distribution of Correlation Coefficients at Critical Strength: '+str(strength))
            plt.xlabel('Correlation Coefficient (R)')
            plt.ylabel('Number of Occurrences')
            plt.plot(Rrange, yR)
            plt.savefig(f'{foldername}/Distribution of Correlation Coefficients at Critical Strength:{str(strength)}.png')
        else:
            plt.title('Distribution of Correlation Coefficients at Strength: ' + str(strength))
            plt.xlabel('Correlation Coefficient (R)')
            plt.ylabel('Number of Occurrences')
            plt.plot(Rrange, yR)
            plt.savefig(f'{foldername}/Distribution of Correlation Coefficients at Strength:{str(strength)}.png')
            #plt.show()
    v=([rmean, (sigma, gsigma)])
    r.append(v[0])
    error.append(v[1][1])
    ostrengths.append(strength)
    grms.append(rms)
    return rmean, (sigma, gsigma)


#getcorrelation('20220104-FRB180814.J422+73-T1.csv', 10, 250, 1250, 1 / 1200, strength=5, plot1=True, plot2=True, showrs=True, printoutput=True, histogram=True)
#plt.show()

def find5S(datafile,Detector,Nstrengths,cores,trials,A,B,dt,histogram=False,plot1=False,plot2=False,crithistogram=False,savedata=True):
    allstrengths=np.linspace(0,10,Nstrengths)
    #strengths=list(x)
    if Nstrengths%cores==0:
        divs=Nstrengths/cores
    else:
        divs=Nstrengths//cores+1
    #r=[]
    if __name__ == '__main__':
        foldername = 'run_' + str(datetime.datetime.now())[:10] + '_' + str(datetime.datetime.now())[11:16]
        os.mkdir(foldername, mode=0o777)
        #error = []
        with Manager() as manager:
            r = manager.list()
            error = manager.list()
            ostrengths=manager.list()
            grms=manager.list()
            splitstrengths=np.array_split(allstrengths,divs)
            for strengths in tqdm(splitstrengths):
                arguments = []
                processes = []
                for i in tqdm(strengths):
                    arguments.append([datafile, Detector, trials, A, B, dt, i, False, False, False, False, histogram, False, r, error, ostrengths, grms, foldername])
                for i in range(len(strengths)):
                    p = multiprocessing.Process(target=getcorrelation, args=arguments[i])
                    p.start()
                    processes.append(p)
                for process in processes:
                    process.join()
            print(grms)
            def model1(t,A):
                return 1-(1/(np.e**(A*t)))
            sstrengths=np.linspace(0,10,1000)
            best_params1, cov_matrix1 = so.curve_fit(model1, xdata = ostrengths, ydata = r, p0 = [1])
            if plot1==True:
                plt.figure()
                plt.plot(ostrengths,r,'.')
                plt.errorbar(ostrengths,r, capsize=10, yerr=error, fmt='none')
                plt.title('Strength of Correlation at Different Strengths of Injected Signal')
                plt.xlabel('Relative Strength of Injected Signal')
                plt.ylabel('Average Correlation Coefficient (4 Trials)')
                plt.plot(sstrengths, model1(sstrengths, best_params1[0]), 'r-', label = 'Fit')
                plt.savefig(f'{foldername}/Strength of Correlation at Different Strengths of Injected Signal')
            fity=np.array(r)/np.array(error)

            def model(t,A,B):
                return A*t**B

            best_params, covariant_matrix = so.curve_fit(model, xdata = ostrengths, ydata = fity, p0 = [1,1])
            xyerror=np.sqrt(np.diagonal(covariant_matrix))

            S5=5*np.ones(1000)

            if plot2==True:
                plt.figure()
                plt.plot(ostrengths,fity,'.')
                plt.xlabel('Strength')
                plt.ylabel('R To Error Ratio')
                plt.title('R To Error Ratio At Range Of Strengths')
                plt.plot(sstrengths, model(sstrengths, best_params[0], best_params[1]), 'r-', label = 'Fit')
                bound_lower = model(sstrengths, *(best_params - xyerror))
                bound_upper = model(sstrengths, *(best_params + xyerror))
                plt.fill_between(sstrengths, bound_lower, bound_upper,color='black', alpha=0.15)
                plt.plot(sstrengths,S5, label = 'S5')
                plt.legend()
                plt.savefig(f'{foldername}/R To Error Ratio')

            crits=(5/best_params[0])**(1/best_params[1])
            critslow=(5 / (best_params[0] + xyerror[0])) ** (1 / (best_params[1] + xyerror[1]))
            critshigh=(5/(best_params[0]-xyerror[0]))**(1/(best_params[1]-xyerror[1]))
            critsv=crits*grms[0]
            critr=model1(crits,best_params1[0])
            critrlow=model1(critslow,best_params1[0])
            critrhigh=model1(critshigh,best_params1[0])
            critslowv=critslow*grms[0]
            critshighv=critshigh*grms[0]
            if crithistogram==True:
                h = getcorrelation(datafile ,Detector, trials, A, B, dt, strength=crits, histogram=True,crit=True)
                #plt.savefig(f'{foldername}/Distribution of Correlation Coefficients at Critical Strength:' + str(crits) + '.png')

            if histogram==True or plot1==True or plot2==True or crithistogram==True:
                plt.show()

            if savedata==True:
                result='\n\n------------------------------------------------------------------------------------------\n\n'+'Data File: '+datafile+', Detector Number: '+str(Detector)+', Duration of ECM Data Used: '+str(B-A)+'s'+', Time Resolution: '+str(dt)+'s'+', Number of Sampled Strengths: '+str(Nstrengths)+', Number of Rs per Gaussian Fit: '+str(trials)+'\n\n'+'---->Critical Strength in RMS: Lower Bound:'+str(critslow)+'; Estimate: '+str(crits)+'; Upper Bound: '+str(critshigh)+'\nCritical Strength in Volts: Lower Bound:'+str(critslowv)+'; Estimate:'+str(critsv)+'; Upper Bound:'+str(critshighv)+'\nCritical R: Lower Bound: '+str(critrlow)+'; Estimate: '+str(critr)+'; Upper Bound: '+str(critrhigh)
                with open('Results.txt', 'a') as f:
                    f.writelines(result)
                f.close()
                with open(f'{foldername}/Results.txt', 'a') as f:
                    f.writelines(result)
                f.close()
            return ([crits,(critslow,critshigh)],critsv,critr)

#tdata=[]
#mdata=[]

#x=find5S('20220104-FRB20201124A-T4.csv',1,11,10,180,280,1/2400,histogram=True,plot1=True,plot2=True,crithistogram=False,savedata=True)
x=find5S('20220104-FRB180814.J422+73-T1.csv',1,21,4,250,500,800,1/1200,histogram=True,plot1=True,plot2=True,crithistogram=False,savedata=True)
print(x)

print('xx')


#plt.plot(tdata,mdata)