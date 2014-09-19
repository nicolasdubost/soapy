import numpy
import numpy.random
import logging
import scipy.ndimage
import scipy.optimize

from . import AOFFT, aoSimLib, zerns
from .opticalPropagationLib import angularSpectrum

#xrange now just "range" in python3. 
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range



class WFS(object):
    ''' A  WFS class.

        This is a base class which contains methods to initialise the WFS,
        and calculate the phase across the WFSs input aperture, given the WFS
        guide star geometry. 

    '''

    def __init__(self, subaps, pupilSize, GSPos,
                 pxlScale, scrnHeights, mask, subapThreshold,
                 GSHeight, pxlsPerSubap, subapOversamp, subapFOV,
                 waveLength, telDiam, LGS, elongation,
                 elongLayers, fftProcs, pyfftw_FLAG, pyfftw_THREADS,
                 lgsLaunchPos, phsWvl, removeTT, SNR, AEN, bitDepth, 
                 propMethod, scrns):

        self.subaps = subaps
        self.pupilSize = pupilSize
        self.centMethod = "simple"
        self.GSPos = GSPos
        self.pxlScale = pxlScale
        self.scrnHeights = scrnHeights
        self.subapThreshold = subapThreshold
        self.GSHeight = GSHeight
        self.mask = mask
        self.pxlsPerSubap = pxlsPerSubap
        self.subapOversamp = subapOversamp
        self.subapFOV = subapFOV
        self.subapFOVrad = subapFOV * numpy.pi / (180. * 3600)
        self.waveLength = waveLength
        self.telDiam = telDiam
        self.subapDiam = self.telDiam/self.subaps
        self.LGS = LGS
        self.elong = elongation
        self.elongLayers = elongLayers
        self.lgsLaunchPos = lgsLaunchPos
        self.removeTT = removeTT
        self.SNR = SNR
        self.bitDepth = bitDepth
        self.angleEquivNoise = AEN * float(pxlsPerSubap)/subapFOV
        
        #Choose propagation method
        print("WFS Propagation Method:",propMethod)
        if propMethod=="physical" or propMethod=="phys":
            self.WfsPhase = self.WfsPhasePhysical
        else:
            self.WfsPhase = self.WfsPhaseGeo
            
        
        #Centroiding threshold (as fraction of max value in subap)
        self.cent_threshold = 0.2

        self.iMat=False

        #Initial set of phase screens
        self.scrns=scrns

        #Phase power scaling factor for wfs wavelength
        self.r0Scale = phsWvl/self.waveLength

        #spacing between subaps in pupil Plane (size "pupilSize")
        self.PPSpacing = float(self.pupilSize)/self.subaps

        #Spacing on the "FOV Plane" - the number of elements required
        #for the correct subap FOV (from way FFT "phase" to "image" works)
        self.subapFOVSpacing = numpy.round(self.subapDiam * self.subapFOVrad / self.waveLength)

        #make twice as big to double subap FOV
        self.SUBAP_OVERSIZE = 2
        self.detectorPxls = self.pxlsPerSubap*self.subaps
        self.subapFOVSpacing *= self.SUBAP_OVERSIZE
        self.pxlsPerSubap2 = self.SUBAP_OVERSIZE*self.pxlsPerSubap

        #LGS Elongation##############################
        #If LGS find meta pupil sizes for each phase screen
        elongSpacing = float(self.elong)/(self.elongLayers-1)
        self.elongHeights = numpy.arange(self.GSHeight-self.elong/2.,
                        self.GSHeight+self.elong/2.+elongSpacing,
                           elongSpacing)


        if self.elong!=0:
            self.elongZs = zerns.zernikeArray([1,2,3], self.pupilSize)
            self.elongRadii={}
            self.elongPhaseAdditions = numpy.empty( (self.elongLayers,
                                            self.pupilSize, self.pupilSize))
            for i in xrange(self.elongLayers):
                self.elongRadii[i] = self.findMetaPupilSize(
                                                    float(self.elongHeights[i]))
                self.elongPhaseAdditions[i] = self.elongPhaseAddition(i)



        if self.GSHeight!=0:
            self.radii = self.findMetaPupilSize(self.GSHeight)
        else:
            self.radii=None

        self.findActiveSubaps()
        #self.scaledMask = numpy.round(scipy.ndimage.zoom(self.mask,
                                     #self.subapFOVSpacing/self.PPSpacing))
        self.scaledMask = aoSimLib.zoom(self.mask,
                                        self.subapFOVSpacing*self.subaps)
        
        #The data type of the arrays
        CDTYPE = numpy.complex64
        DTYPE = numpy.float32
        
        #Initialise FFT classes
        #########################################
        self.subapFFTPadding = self.pxlsPerSubap2 * self.subapOversamp
        if self.subapFFTPadding < self.subapFOVSpacing:
            while self.subapFFTPadding<self.subapFOVSpacing:
                self.subapOversamp+=1
                self.subapFFTPadding\
                        =self.pxlsPerSubap2*self.subapOversamp

            logging.warning("requested WFS FFT Padding less than FOV size... Setting oversampling to: %d"%self.subapOversamp)

        if fftProcs==0 or fftProcs>1:
            self.FFT = AOFFT.mpFFT(inputSize=(self.activeSubaps,
                                            self.subapFFTPadding,
                                            self.subapFFTPadding),
                                axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                            fftw_FLAGS=(pyfftw_FLAG,),processes=fftProcs)
        else:
            self.FFT = AOFFT.FFT(inputSize=(self.activeSubaps,
                                        self.subapFFTPadding,
                                        self.subapFFTPadding),
                            axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                            THREADS=pyfftw_THREADS, fftw_FLAGS=(pyfftw_FLAG,))

        if self.LGS!=None:
            self.iFFT = AOFFT.FFT(inputSize = (self.activeSubaps,
                                        self.subapFFTPadding,
                                        self.subapFFTPadding),
                            axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                            THREADS=pyfftw_THREADS,fftw_FLAGS=(pyfftw_FLAG,))

            self.lgs_iFFT = AOFFT.FFT(inputSize = (self.subapFFTPadding,
                                                self.subapFFTPadding),
                            axes=(0,1), mode="pyfftw",dtype=CDTYPE,
                            THREADS=pyfftw_THREADS,fftw_FLAGS=(pyfftw_FLAG,))

        ######################################################

        #Tilt Correction for even number sizes subaps -
        #otherwise the PSF is binned to one corner
        self.optimiseTilt()

        #Tell LGS some WFS stuff
        if self.LGS:
            self.LGS.setWFSParams(self.SUBAP_OVERSIZE*self.subapFOVrad,
                                     self.subapOversamp, self.subapFFTPadding)
            

        #initialise some arrays which will be useful later
        
        #First make wfs detector array - only needs to be of ints,
        #Find which kind to save memory
        if bitDepth==8 or bitDepth==16 or bitDepth==32:
            self.dPlaneType = eval("numpy.uint%d"%bitDepth)
        else:
            self.dPlaneType = numpy.uint32
        self.maxFlux = 0.7 * 2**bitDepth -1
        self.wfsDetectorPlane = numpy.zeros( (  self.detectorPxls,
                                                self.detectorPxls   ),
                                               dtype = self.dPlaneType )

        self.wfsPhase = numpy.zeros( [self.pupilSize]*2 )
        self.scaledWfsPhase = numpy.zeros([ self.subaps*self.subapFOVSpacing]*2)
        self.subapArrays=numpy.empty((self.activeSubaps,
                                      self.subapFOVSpacing,
                                      self.subapFOVSpacing),
                                     dtype=CDTYPE)
        self.binnedFPSubapArrays = numpy.zeros( (self.activeSubaps,
                                                self.pxlsPerSubap2,
                                                self.pxlsPerSubap2),
                                                dtype=DTYPE)
        self.FPSubapArrays = numpy.zeros((self.activeSubaps,
                                          self.subapFFTPadding,
                                          self.subapFFTPadding),dtype=DTYPE)
        
        #For Fresnel Propagation
        self.Efield = numpy.zeros((self.pupilSize, self.pupilSize),
                                                        dtype=CDTYPE)


    def optimiseTilt(self):
        '''
        Finds the optimimum correction tilt to apply to each
        subap to account for an odd even number of points in the subap
        FFT.
        to be executed during WFS initialisation.
        '''
        spacing = 1./self.subapFOVSpacing
        X,Y = numpy.meshgrid(numpy.arange( -1+spacing,1+spacing,2*spacing),
                             numpy.arange(-1+spacing,1+spacing,2*spacing) )
       
        O =numpy.ones( (self.subapFOVSpacing, self.subapFOVSpacing) )
        self.YTilt = Y

        res = scipy.optimize.minimize(self.optFunc,0,args=(X,O),tol=0.1,
                                options={"maxiter":6})
        A = res["x"]
        self.XTilt = A*X
        self.YTilt = A*Y

    def optFunc(self,A,X,O):
        '''
        Function the <optimiseTilt> function uses to optimise
        corrective subap tilt
        '''
        self.XTilt = A*X
        slopes = self.oneSubap(O)

        return abs(slopes[1])

    def oneSubap(self,phs):
        '''
        Processes one subaperture only, with given phase
        '''
        eField = numpy.exp(1j*phs)
        FP = numpy.abs(numpy.fft.fftshift(
            numpy.fft.fft2(eField * numpy.exp(1j*self.XTilt),
                s=(self.subapFFTPadding,self.subapFFTPadding))))**2

        FPDetector = aoSimLib.binImgs(FP,self.subapOversamp)

        slope = aoSimLib.simpleCentroid(FPDetector, self.cent_threshold)
        slope -= self.pxlsPerSubap2/2.
        return slope



    def findMetaPupilSize(self,GSHeight):
        '''
        Function to evaluate the sizes of the effective metePupils
        at each screen height if an GS of finite height is used.
        '''

        radii={}

        for i in self.scrns:
            #Find radius of metaPupil geometrically (fraction of pupil at
            # Ground Layer)
            radius = (self.telDiam/2.) * (
                                1-(float(self.scrnHeights[i])/GSHeight))
            radii[i]= numpy.round(radius * self.pxlScale)

            #If scrn is above LGS, radius is 0
            if self.scrnHeights[i]>=GSHeight:
                radii[i]=0

        return radii

    def findActiveSubaps(self):
        '''
        Finds the subapertures which are not empty space
        determined if mean of subap coords of the mask is above threshold
        '''

        self.subapCoords = aoSimLib.findActiveSubaps(self.subaps,self.mask, self.subapThreshold)
        self.activeSubaps = self.subapCoords.shape[0]

        #When scaled to pxl sizes, need to scale subap coordinates too!
        self.detectorSubapCoords = numpy.round(
            self.subapCoords*(self.detectorPxls/float(self.pupilSize) ) )


    def elongPhaseAddition(self,elongLayer):

        #Calculate the path difference between the central GS height and the
        #elongation "layer"
        #Define these to make it easier
        h = self.elongHeights[elongLayer]
        dh = h-self.GSHeight
        H = self.GSHeight
        d = self.lgsLaunchPos * self.telDiam/2.
        D = self.telDiam
        theta = (d.astype("float")/H) - self.GSPos

        #for the focus terms....
        focalPathDiff = (2*numpy.pi/self.waveLength) * ( (
            ( (self.telDiam/2.)**2 + (h**2) )**0.5\
          - ( (self.telDiam/2.)**2 + (H)**2 )**0.5 ) - dh )

        #For tilt terms.....
        tiltPathDiff = (2*numpy.pi/self.waveLength) * (
            numpy.sqrt( (dh+H)**2. + ( (dh+H)*theta-d-D/2.)**2 )
            + numpy.sqrt( H**2 + (D/2. - d + H*theta)**2 )
            - numpy.sqrt( H**2 + (H*theta - d - D/2.)**2)
            - numpy.sqrt( (dh+H)**2 + (D/2. - d + (dh+H)*theta )**2 )    )


        phaseAddition = numpy.zeros( (  self.pupilSize, self.pupilSize) )

        phaseAddition +=( (self.elongZs[2]/self.elongZs[2].max())
                             * focalPathDiff )
        #X,Y tilt
        phaseAddition += ( (self.elongZs[0]/self.elongZs[0].max())
                            *tiltPathDiff[0] )
        phaseAddition += ( (self.elongZs[1]/self.elongZs[1].max())
                            *tiltPathDiff[1])


        return phaseAddition


    def metaPupilPos(self,height):
        '''
        Finds the centre of a metapupil at a given height, when offset by a given angle in arcsecs
        '''

        #Convert positions into radians
        GSPos=numpy.array(self.GSPos)*numpy.pi/(3600.0*180.0)

        #Position of centre of GS metapupil off axis at required height
        GSCent=numpy.tan(GSPos)*height

        return GSCent



    def metaPupilPhase(self,scrn,height, radii=None):
        '''
        Returns the phase across a metaPupil 
        at some height and angular offset in arcsec.
        Interpolates if cone effect is required
        '''

        GSCent = self.metaPupilPos(height) * self.pxlScale
        scrnX,scrnY=scrn.shape

        if (scrnX/2.+GSCent[0]-self.pupilSize/2.0)<0 \
           or (scrnX/2.+GSCent[0]-self.pupilSize/2.0) \
           >scrnX or scrnX/2.+GSCent[0]-self.pupilSize/2.0 \
           <0 or (scrnY/2.+GSCent[1]-self.pupilSize/2.0) \
           >scrnY :

            raise ValueError( "GS seperation requires larger screen size" )

        metaPupil=scrn[ int(round(scrnX/2.+GSCent[0]-self.pupilSize/2.0)):
                            int(round(scrnX/2.+GSCent[0]+self.pupilSize/2.0)),
                        int(round(scrnY/2.+GSCent[1]-self.pupilSize/2.0)):
                            int(round(scrnY/2.+GSCent[1]+self.pupilSize/2.0))]

        if self.GSHeight!=0:
            if radii!=0:
                metaPupil = aoSimLib.zoom(
                                    metaPupil[
                        int(round(self.pupilSize/2.-radii)):
                        int(round(self.pupilSize/2. +radii)),
                        int(round(self.pupilSize/2.-radii)):
                        int(round(self.pupilSize/2. + radii))],
                                        (self.pupilSize,self.pupilSize))

        return metaPupil


    def WfsPhaseGeo(self, radii=None):
        '''
        Creates the total phase on a wavefront sensor which 
        is offset by a given angle
        '''
        self.wfsPhase[:] = 0
        for i in self.scrns:
            if radii:
                phase = self.metaPupilPhase(self.scrns[i],self.scrnHeights[i],
                        radii[i])
            else:
                phase = self.metaPupilPhase(self.scrns[i],self.scrnHeights[i])
            
            self.wfsPhase += phase
            
        self.EField = numpy.exp(1j*self.wfsPhase)


    def WfsPhasePhysical(self, radii=None, initialEField=None):
        '''
        Finds total WFS complex amplitude by propagating light down
        phase scrns'''

        scrnNo = len(self.scrns)-1
        ht = self.scrnHeights[scrnNo]
        delta = (self.pxlScale)**-1. #Grid spacing
        
        #If initialEField, use that
        if initialEField==None:
            self.EField[:] = self.mask
        else:
            self.EField = initialEField
        
        #Get initial Phase for highest scrn and turn to efield
        if radii:
            phase1 = self.metaPupilPhase(self.scrns[scrnNo], ht, radii[scrnNo])
        else:
            phase1 = self.metaPupilPhase(self.scrns[scrnNo], ht)
        
        self.EField *= numpy.exp(1j*phase1)
        
        #Loop through remaining scrns in reverse order - update ht accordingly
        for i in xrange(scrnNo)[::-1]:
            #Get propagation distance for this layer
            z = ht - self.scrnHeights[i]
            ht -= z
            #Do ASP for last layer to next
            self.EField = angularSpectrum(self.EField, self.waveLength, delta,
                            delta, z )
            
            #Get phase for this layer
            if radii:
                phase = self.metaPupilPhase(self.scrns[i],self.scrnHeights[i],
                        radii[i])
            else:
                phase = self.metaPupilPhase(self.scrns[i],self.scrnHeights[i])
            
            #Add add phase from this layer
            self.EField *= numpy.exp(1j*phase)
        
        #If not already at ground, propagate the rest of the way.
        if self.scrnHeights[0]!=0:
            angularSpectrum(self.EField, self.waveLength, delta, delta, ht)


    def readNoise(self, dPlaneArray):

        '''Read Noise'''
       
        dPlaneArray += numpy.random.normal( (self.maxFlux/self.SNR),
        0.1*self.maxFlux/self.SNR, dPlaneArray.shape).clip(0,self.maxFlux).astype(self.dPlaneType)


    def photonNoise(self):

        '''Photon Noise'''
        pass


    def iMatFrame(self,phs):
        '''
        Runs an iMat frame - essentially gives slopes for given "phs",
        useful for other stuff too!
        '''
        self.iMat=True
        #Set "removeTT" to false while we take an iMat
        removeTT = self.removeTT
        self.removeTT=False

        self.FPSubapArrays[:] = 0
        self.EField = numpy.exp(1j*phs)
        self.WfsFocalPlane()
        self.makeFocalPlane()
        self.calculateSlopes()
        
        self.removeTT = removeTT
        self.iMat=False
        
        return self.slopes


    def frame(self,scrns,correction=None):
        '''
        Runs one WFS frame
        '''

        scrns
        #Scale phase to WFS wvl
        for i in xrange(len(scrns)):
            self.scrns[i] = scrns[i].copy()*self.r0Scale

        self.FPSubapArrays[:] = 0

        if self.elong==0:

            self.WfsPhase(self.radii)
            self.uncorrectedPhase = self.wfsPhase.copy()
            if correction!=None:
                self.EField *= numpy.exp(-1j*correction)
            self.WfsFocalPlane()


        if self.elong!=0:

            for i in xrange(self.elongLayers):

                self.WfsPhase(self.elongRadii[i])
                self.uncorrectedPhase = self.wfsPhase
                #self.wfsPhase += self.elongPhaseAdditions[i]
                self.EField *= numpy.exp(1j*self.elongPhaseAdditions[i])
                if correction!=None:
                    self.EField *= numpy.exp(-1j*correction)
                self.WfsFocalPlane()

        self.makeFocalPlane()
        #Any noise must go in here.

        self.calculateSlopes()

        return self.slopes

class ShackHartmannWfs(WFS):
    """Class to simulate a Shack-Hartmann WFS"""

    def photonNoise(self):

        '''Photon Noise'''
        pass


    def LGSUplink(self):
        '''
        A method to deal with convolving the LGS PSF
        with the subap focal plane.
        '''


        self.lgs_iFFT.inputData[:] = self.LGS.PSF
        self.iFFTLGSPSF = self.lgs_iFFT()

        self.iFFT.inputData[:] = self.FPSubapArrays
        self.iFFTFPSubapsArray = self.iFFT()

        #Do convolution
        self.iFFTFPSubapsArray *= self.iFFTLGSPSF

        #back to Focal Plane.
        self.FFT.inputData[:] = self.iFFTFPSubapsArray
        self.FPSubapArrays[:] = numpy.fft.fftshift( self.FFT(), axes=(1,2)).real

        # self.FPSubapArrays[:] = (numpy.fft.fftshift(
        #             numpy.fft.fft2( self.iFFTFPSubapsArray), axes=(1,2))).real
        #       
        self.FPSubapArrays[:] = self.FPSubapArrays[:,::-1,::-1]

    def calculateSlopes(self):
        '''
        returns wfs slopes from wfsFocalPlane
        '''

        #Sort out FP into subaps
        subapArrays = numpy.empty( (self.activeSubaps,  self.pxlsPerSubap,
                                                        self.pxlsPerSubap) )

        for i in xrange(self.activeSubaps):
            x,y = self.detectorSubapCoords[i]
            x = int(x)
            y = int(y)
            subapArrays[i] = self.wfsDetectorPlane[ x:x+self.pxlsPerSubap,
                                                    y:y+self.pxlsPerSubap ]

        #Use self made function which processes all subaps at same time
        slopes=aoSimLib.simpleCentroid(subapArrays,self.cent_threshold)
        #shift slopes relative to subap centre
        slopes-=self.pxlsPerSubap/2.0
        
        if self.removeTT:
            slopes = (slopes.T - slopes.mean(1)).T
        
        self.slopes=slopes.reshape(self.activeSubaps*2)
        
        if self.angleEquivNoise and not self.iMat:
            self.slopes += numpy.random.normal(0,self.angleEquivNoise, 
                                                        2*self.activeSubaps)
        
        return self.slopes



    def WfsFocalPlane(self):
        '''
        Calculates the wfs focal plane, given the phase across the WFS
        '''
        #create an array of individual subap EFields
        # self.scaledWfsPhase = aoSimLib.zoom( self.wfsPhase,
                                    # self.subaps*self.subapFOVSpacing)
        
        self.scaledEField = aoSimLib.zoom( self.EField, 
                self.subaps*self.subapFOVSpacing) * self.scaledMask 

        for i in xrange(self.activeSubaps):
            x,y = numpy.round(self.subapCoords[i] *
                                     self.subapFOVSpacing/self.PPSpacing)
            self.subapArrays[i] = self.scaledEField[int(x):
                                    int(x+self.subapFOVSpacing) ,
                                              int(y):
                                              int(y+self.subapFOVSpacing)]

        #do the fft to all subaps at the same time
        # and convert into intensity
        self.FFT.inputData[:] = 0
        self.FFT.inputData[:,:int(round(self.subapFOVSpacing))
                        ,:int(round(self.subapFOVSpacing))] \
                = self.subapArrays*numpy.exp(1j*(self.XTilt+self.YTilt))

        self.FPSubapArrays += numpy.abs(numpy.fft.fftshift(self.FFT(),
                                        axes=(1,2)))**2



    def makeFocalPlane(self):
        '''
        if required, will convolve final psf with LGS psf, then bins
        psf down to detector size. Finally puts back into wfsFocalPlane Array
        in correct order.
        '''

        #If required, convolve with LGS PSF
        if self.LGS != None and self.iMat!=True:
            self.LGSUplink()

        #bins back down to correct size and then
        #fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = aoSimLib.binImgs(self.FPSubapArrays,
                                                        self.subapOversamp)

        self.binnedFPSubapArrays = self.maxFlux* (self.binnedFPSubapArrays.T/
                                            self.binnedFPSubapArrays.max((1,2))
                                                                         ).T

        self.wfsDetectorPlane[:] = 0

        for i in xrange(self.activeSubaps):

            x,y=self.detectorSubapCoords[i]

            #Set default position to put arrays into (2 subap FOV)
            x1 = int(round(x-self.pxlsPerSubap/2.))
            x2 = int(round(x+self.pxlsPerSubap*3./2))
            y1 = int(round(y-self.pxlsPerSubap/2.))
            y2 = int(round(y+self.pxlsPerSubap*3./2.))

            #Set defualt size of input array (i.e. all of it)
            x1_fp = int(0)
            x2_fp = int(round(self.pxlsPerSubap2))
            y1_fp = int(round(0))
            y2_fp = int(round(self.pxlsPerSubap2))

            #If at the edge of the field, may only fit a fraction in 
            if x==0:
                x1 = int(0)
                x1_fp = int(round(self.pxlsPerSubap/2.))

            elif x==(self.detectorPxls-self.pxlsPerSubap):
                x2 = int(round(self.detectorPxls))
                x2_fp = int(round(-1*self.pxlsPerSubap/2.))

            if y==0:
                y1 = int(0)
                y1_fp = int(round(self.pxlsPerSubap/2.))

            elif y==(self.detectorPxls-self.pxlsPerSubap):
                y2 = int(self.detectorPxls)
                y2_fp = int(round(-1*self.pxlsPerSubap/2.))

            self.wfsDetectorPlane[x1:x2, y1:y2] += (
                    self.binnedFPSubapArrays[i, x1_fp:x2_fp, y1_fp:y2_fp] )

        if self.SNR and self.iMat!=True:

            self.photonNoise()
            self.readNoise(self.wfsDetectorPlane)


def pyrimid(phs):

    shapeX,shapeY = phs.shape
    quads = numpy.empty( (4,shapeX/2.,shapeX/2.), dtype="complex64")
    EField = numpy.exp(1j*phs) * aoSimLib.circle(shapeX/2,shapeX)
    FP = numpy.fft.fftshift(numpy.fft.fft2(EField))
    n=0
    for x in xrange(2):
        for y in xrange(2):
            quads[n] = FP[x*shapeX/2 : (x+1)*shapeX/2,
                        y*shapeX/2 : (y+1)*shapeX/2]
            n+=1

    PPupilPlane = abs(numpy.fft.ifft2(quads))**2

    allPupilPlane = numpy.empty( (shapeX,shapeY) )
    allPupilPlane[:shapeX/2.,:shapeY/2] = PPupilPlane[0]
    allPupilPlane[shapeX/2.:,:shapeY/2] = PPupilPlane[1]
    allPupilPlane[:shapeX/2.,shapeY/2:] = PPupilPlane[2]
    allPupilPlane[shapeX/2.:,shapeY/2:] = PPupilPlane[3]

    return allPupilPlane


