import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# nice plots
import cmasher as cmr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

projection = ccrs.EqualEarth(central_longitude=205)
transform = ccrs.PlateCarree()

cmap = mpl.colormaps['inferno_r']
cmapdiff = cmr.fusion_r
cmapper = mpl.colormaps['inferno']
cmapSST = mpl.colormaps['RdBu_r']
cmapinc = cmr.sunburst_r

cmapdiff_oneway = cmapdiff(np.linspace(0.5,1,cmapdiff.N))
cmapdiff_oneway = mpl.colors.LinearSegmentedColormap.from_list('onewayfusion',cmapdiff_oneway)

boundsacc = 100*np.arange(0.33,0.465,0.005)
boundsdiff = 100*np.arange(-0.05,0.052,0.002)

boundsSDP = 100*np.arange(0.33,0.61,0.01)
boundsdiffSDP = 100*np.arange(-0.2,0.21,0.01)

boundsinc = 100*np.arange(0,0.155,0.005)
boundsinc40 = 100*np.arange(0,0.37,0.02)

boundsobsera = 100*np.arange(0.33,0.85,0.01)
boundsdiffobsera = 100*np.arange(-0.3,0.32,0.02)

boundsdiff_oneway = 100*np.arange(0,0.105,0.005)
boundsdiff_onewaySDP = 100*np.arange(0,0.205,0.005)
normdiff_oneway = colors.BoundaryNorm(boundaries=boundsdiff_oneway, ncolors=cmapdiff_oneway.N)
normdiff_onewaySDP = colors.BoundaryNorm(boundaries=boundsdiff_onewaySDP, ncolors=cmapdiff_oneway.N)


normacc = colors.BoundaryNorm(boundaries=boundsacc, ncolors=cmap.N)
normdiff = colors.BoundaryNorm(boundaries=boundsdiff, ncolors=cmapdiff.N)

normaccSDP = colors.BoundaryNorm(boundaries=boundsSDP, ncolors=cmap.N)
normdiffSDP = colors.BoundaryNorm(boundaries=boundsdiffSDP, ncolors=cmapdiff.N)

normaccobsera = colors.BoundaryNorm(boundaries=boundsobsera, ncolors=cmap.N)
normdiffobsera = colors.BoundaryNorm(boundaries=boundsdiffobsera, ncolors=cmapdiff.N)

norminc = colors.BoundaryNorm(boundaries=boundsinc, ncolors=cmapinc.N)
norminc40 = colors.BoundaryNorm(boundaries=boundsinc40, ncolors=cmapinc.N)

boundsclassfreq = np.arange(0,1.05,0.05)
normclassfreq = colors.BoundaryNorm(boundaries=boundsclassfreq,ncolors=cmap.N)

boundsclassper = 100*np.arange(0,1.05,0.05)
normclassper = colors.BoundaryNorm(boundaries=boundsclassper,ncolors=cmap.N)

boundstrend = np.arange(-1.1,1.15,0.05)
normtrend = colors.BoundaryNorm(boundaries=boundstrend,ncolors=cmapSST.N)

continents = 'xkcd:grey'

def compareaccuracy(acctest,shuffletest,lon,lat):
 
    plt.figure(figsize=(7,9))

    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,0],cmap=cmap,norm=normacc,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('full model')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*shuffletest[:,:,0],cmap=cmap,norm=normacc,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('shuffle model')
    
    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,65,5))
    cbar.ax.set_ylabel('accuracy (%)')

    diffplot = acctest[:,:,0]-shuffletest[:,:,0]
    # diffplot[diffplot<0] = 0    
    
    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(diffplot),cmap=cmapdiff,norm=normdiff,transform=transform)
#     a3.scatter(lonsig_IV,latsig_IV,marker='.',color='grey',transform=transform)
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('accuracy difference')
    
    cax3=plt.axes((0.89,0.12,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-10,15,5))
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    
    # plt.savefig("figures/compIV.png",dpi=200)

def compareaccuracy_null(acctest,null,lon,lat):
 
    plt.figure(figsize=(7,9))

    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,0],cmap=cmap,norm=normacc,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('full model')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,null,cmap=cmap,norm=normacc,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('null model')

    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,65,5))
    cbar.ax.set_ylabel('accuracy (%)')
    
    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(acctest[:,:,0])-null,cmap=cmapdiff,norm=normdiff,transform=transform)
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('accuracy difference')

    cax3=plt.axes((0.89,0.11,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-15,20,5))
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    
    # plt.savefig("figures/compIV.png",dpi=200)
    
def compareaccuracy_SDP(acctest,lon,lat):
 
    plt.figure(figsize=(7,9))

    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,0],cmap=cmap,norm=normaccSDP,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('all predictions')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*acctest[:,:,8],cmap=cmap,norm=normaccSDP,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('20% most confident')
    
    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,75,5))
    cbar.ax.set_ylabel('accuracy (%)')
    
    diffplot = acctest[:,:,8]-acctest[:,:,0]
    # diffplot[diffplot`<0] = 0

    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(diffplot),cmap=cmapdiff,norm=normdiffSDP,transform=transform)
#     a3.scatter(lonsig_SDP,latsig_SDP,marker='.',color='grey',transform=transform)
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('accuracy increase')
    
    cax3=plt.axes((0.89,0.11,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-15,20,5))
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    # plt.savefig("figures/compSDP.png",dpi=200)
    # 
def compareaccuracy_IVSDP(acctest,shuffletest,lon,lat):
 
    plt.figure(figsize=(7,9))

    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,8],cmap=cmap,norm=normaccSDP,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('full model, 20% most confident')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*shuffletest[:,:,8],cmap=cmap,norm=normaccSDP,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('shuffle model, 20% most confident')

    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,75,5))
    cbar.ax.set_ylabel('accuracy (%)')

    diffplot = acctest[:,:,8]-shuffletest[:,:,8]
    # diffplot[diffplot<0] = 0

    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(diffplot),cmap=cmapdiff,norm=normdiffSDP,transform=transform)
#     a3.scatter(lonsig_IVSDP,latsig_IVSDP,marker='.',color='grey',transform=transform)
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('difference')

    cax3=plt.axes((0.89,0.11,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-15,20,5))
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    # plt.savefig("figures/compIVSDP.png",dpi=200)

def getmeanmetric(dsvar,seedchoose,nlat,nlon):
    
    dsvar = np.asarray(dsvar)
    shapegrab = seedchoose.shape
    
    if len(dsvar.shape)>3:
        shapegrabvar = (shapegrab[0],shapegrab[1],shapegrab[2],dsvar.shape[3])
    else:
        shapegrabvar = (shapegrab[0],shapegrab[1],shapegrab[2])
    
    vartest = np.empty(shapegrabvar)
    
    for ilat in range(nlat):
        for ilon in range(nlon):
            for isel in range(shapegrab[2]):
                vartest[ilat,ilon,isel,] = dsvar[ilat,ilon,seedchoose[ilat,ilon,isel],]
    
    vartest = np.mean(vartest,axis=2)
    
    return vartest

def classfreqplot(freqclasstest,lon,lat):
    
    plt.figure(figsize=(12,5))
    nclasses = 4
    #classes = [0,1,2,5]
    classstr = ['EF and IV agree', 'EF and IV disagree, IV wins','EF and IV disagree, EF wins','Neither EF nor IV wins']
    
    for i in range(nclasses):
        freqplot = freqclasstest[:,:,i]
        a1=plt.subplot(2,2,i+1,projection=projection)
        c1=a1.pcolormesh(lon,lat,freqplot,cmap=cmap,transform=transform,norm=normclassfreq)
        a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
        plt.title(classstr[i])
    
    cax=plt.axes((0.97,0.3,0.02,0.4))
    cbar=plt.colorbar(cax=cax,mappable=c1,ticks=np.arange(0,1.2,0.2))
    cbar.ax.set_ylabel("freq. of prediction type")
    plt.tight_layout()
#     plt.savefig("figures/fourclasses_better.png",dpi=200)
    plt.show()
    
def trendmap(lcutoffmat,ucutoffmat,lon,lat):
    
    plt.figure(figsize=(6,4))
    
    a1=plt.subplot(2,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,lcutoffmat,norm=normtrend,cmap=cmapSST,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('lower cut off')
    
    a2=plt.subplot(2,1,2,projection=projection)
    a2.pcolormesh(lon,lat,ucutoffmat,norm=normtrend,cmap=cmapSST,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('upper cut off')
    
    cax = plt.axes((0.8,0.3,0.02,0.4))
    cbar1 = plt.colorbar(c1,cax=cax,ticks=np.arange(-1,1.5,0.5))
    cbar1.ax.set_ylabel(r"SST trend $^{\circ}$C/decade")
    
    plt.tight_layout()
    plt.savefig("figures/cutoffvals.png",dpi=200)
    plt.show
    
    
def trendmap3(lcutoffmat,ucutoffmat,efmap,lon,lat):
    
    plt.figure(figsize=(6,6))
    
    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,lcutoffmat,norm=normtrend,cmap=cmapSST,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('lower cut off')
    
    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,ucutoffmat,norm=normtrend,cmap=cmapSST,transform=transform)
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('upper cut off')
 
    a3=plt.subplot(3,1,3,projection=projection)
    a3.pcolormesh(lon,lat,efmap,norm=normtrend,cmap=cmapSST,transform=transform)
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('External Forcing trend 2020-2060')
    
    cax = plt.axes((0.82,0.28,0.02,0.4))
    cbar1 = plt.colorbar(c1,cax=cax,ticks=np.arange(-1,1.5,0.5),extend='both')
    cbar1.ax.set_ylabel(r"SST trend $^{\circ}$C/decade")
    
    plt.tight_layout()
    plt.savefig("figures/cutoffvals.png",dpi=200)
    plt.show

def sigcalc(acc,sig,lon,lat):
    
    longrid,latgrid = np.meshgrid(lon,lat)
    
    siglon = longrid[acc<=sig]
    siglat = latgrid[acc<=sig]    
    
    return siglon,siglat

def compareaccuracy_IV_sig(acctest,shuffletest,lon,lat,sig):
    
    siglon,siglat = sigcalc(acctest[:,:,0],sig[:,:,0],lon,lat)
    
    # sigboo = sig[:,:,0]>=acctest[:,:,0]
    diffplot = acctest[:,:,0]-shuffletest[:,:,0]
    # diffplot[diffplot<0] =  0
    
    plt.figure(figsize=(7,9))

    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,0],cmap=cmap,norm=normaccSDP,transform=transform)
    a1.coastlines(color='grey')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('a. full model accuracy')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*shuffletest[:,:,0],cmap=cmap,norm=normaccSDP,transform=transform)
    a2.coastlines(color='grey')
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('c. scrambled IV accuracy')
    
    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,65,5))
    cbar.ax.set_ylabel('accuracy (%)')

    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*diffplot,cmap=cmapdiff_oneway,norm=normdiff_onewaySDP,transform=transform)
    a3.scatter(siglon,siglat,marker='x',color='grey',transform=transform)
    a3.coastlines(color='grey')
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('e. accuracy increase provided by IV')
    
    cax3=plt.axes((0.89,0.12,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(0,25,5),extend='min')
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy (%)')
    
    # plt.savefig("figures/compIV.png",dpi=200)
    
    plt.show()


def compareaccuracy_IVSDP50_sig(acctest,shuffletest,lon,lat,sig):
    
    siglon,siglat = sigcalc(acctest[:,:,5],sig[:,:,5],lon,lat)
    
    # sigboo = sig[:,:,5]>=acctest[:,:,5]
    diffplot = acctest[:,:,5]-shuffletest[:,:,5]
    # diffplot[diffplot<0] = 0
    
    plt.figure(figsize=(7,9))
    
    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,5],cmap=cmap,norm=normaccSDP,transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    a1.coastlines(color='grey')
    plt.title('full model, 50% most confident')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*shuffletest[:,:,5],cmap=cmap,norm=normaccSDP,transform=transform)
    a2.coastlines(color='grey')
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('shuffle, 50% most confident')

    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,75,5))
    cbar.ax.set_ylabel('accuracy (%)')

    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(diffplot),cmap=cmapdiff_oneway,norm=normdiff_onewaySDP,transform=transform)
    a3.scatter(siglon,siglat,marker='x',color='grey',transform=transform)
    a3.coastlines(color='grey')
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('accuracy increase')

    cax3=plt.axes((0.89,0.12,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-0,25,5),extend='min')
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    
    # plt.savefig("figures/compIV.png",dpi=200)
# 
    plt.show()
    
def compareaccuracy_IVSDP_sig(acctest,shuffletest,lon,lat,sig):
    
    siglon,siglat = sigcalc(acctest[:,:,8],sig[:,:,8],lon,lat)
    
    # sigboo = sig[:,:,8]>=acctest[:,:,8]
    diffplot = acctest[:,:,8]-shuffletest[:,:,8]
    # diffplot[diffplot<0] = 0
    
    plt.figure(figsize=(7,9))
    
    a1=plt.subplot(3,1,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,100*acctest[:,:,8],cmap=cmap,norm=normaccSDP,transform=transform)
    a1.coastlines(color='grey')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('b. full model 20% most confident')

    a2=plt.subplot(3,1,2,projection=projection)
    a2.pcolormesh(lon,lat,100*shuffletest[:,:,8],cmap=cmap,norm=normaccSDP,transform=transform)
    a2.coastlines(color='grey')
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('d. scrambled IV 20% most confident')

    cax1=plt.axes((0.89,0.48,0.02,0.3))
    cbar=plt.colorbar(c1,cax=cax1,ticks=np.arange(35,75,5))
    cbar.ax.set_ylabel('accuracy (%)')

    a3=plt.subplot(3,1,3,projection=projection)
    c3=a3.pcolormesh(lon,lat,100*(diffplot),cmap=cmapdiff_oneway,norm=normdiff_onewaySDP,transform=transform)
    a3.scatter(siglon,siglat,marker='x',color='grey',transform=transform)
    a3.coastlines(color='grey')
    a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    plt.title('f. accuracy increase for 20% most confident')

    cax3=plt.axes((0.89,0.12,0.02,0.22))
    cbar3=plt.colorbar(c3,cax=cax3,ticks=np.arange(-0,25,5),extend='min')
    cbar3.ax.set_ylabel(r'$\Delta$ accuracy')
    
    # plt.savefig("figures/compIVSDP.png",dpi=200)
# 
    plt.show()


def compareaccuracy_SDP_allish(acctest,lon,lat):
    
    nplot = 5
    
    titles = ["all predictions", "80% most confident", "60% most confident", "40% most confident", "20% most confident"]
    
    plt.figure(figsize=(10,18))
    
    for iplot in range(nplot):
            
        a1=plt.subplot(5,1,iplot+1,projection=projection)
        c1=a1.pcolormesh(lon,lat,100*acctest[:,:,2*iplot],cmap=cmap,norm=normaccSDP,transform=transform)
        a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
        plt.title(titles[iplot])
    
    # cax=plt.axes((0.2,.04,0.6,0.015))
    # cbar=plt.colorbar(c1,cax=cax,orientation='horizontal',ticks=np.arange(35,65,5))
    # cbar.ax.set_xlabel('accuracy (%)')
    
    plt.tight_layout()
    plt.show()

