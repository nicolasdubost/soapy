import numpy as np
from numpy.fft import fft2,ifft2,fftshift,ifftshift
import aotools
from aotools import wfs,circle

from .. import LGS, logger, lineofsight, interp
from .. import AOFFT, LGS, logger
from . import base
from .. import numbalib

import multiprocessing as mp
from multiprocessing import Process, Queue

# Test
# import pylab as pl
import FITS

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = np.complex64
DTYPE = np.float32


class CAWS(base.WFS):
    """Class to simulate a Calibration and Alignment WFS"""

    def calcInitParams(self):
        """
        Calculate some parameters to be used during initialisation
        Translation between first_simulation.py -> caws.py
        D0              ->  self.los.telescope_diameter
        p               ->  self.wfsConfig.fftOversamp = 3 as default
        wvl             ->  self.wavelength
        wvlBandWidth    ->  self.wfsConfig.wvlBandWidth
        nc              ->  self.pxlsPerLP: input >=4 & >2wvl_max/wvl
        ng              ->  self.nxLP = self.config.nxSubaps
        Np              ->  detectorPxls2 or cropEField.shape[0] (maybe small difference)
        Nt              ->  self.FFTPadding in detector, self.FFTPadding2 in FPlane
        Nsims           ->  self.FFTPaddingList
        ---             ->  self.extensionWindow: pixel padding on true detector
                            around pupil. total padding = self.extensionWindow*2

        TO DO:
                - How does it produce WFS Phase
                - How to to push changes


        """
        super(CAWS, self).calcInitParams()

        # Sort out some required parameters
        self.pxlsPerLP = self.config.pxlsPerSubap
        self.extensionWindow = self.config.extensionWindow
        self.getEF = False # if true, slopes is complex Electric Field instead of phase

        if self.pxlsPerLP%2:
            self.pxlsPerLP += 1
            logger.warning('pxlsPerLP must be an even number. Adding 1')

        m = int(np.ceil(2*(self.wavelength+self.wfsConfig.wvlBandWidth*1e-9/2.)/self.wavelength))
        m = m +m%2
        if self.pxlsPerLP < m:
            logger.warning('pxlsPerLP must be at least %d. Using that value'%(m))
            self.pxlsPerLP = m+0

        # Weird condition. Look in notebook 03/02/2017
        self.nxLP = self.config.nxSubaps
        m = int(np.ceil(1.22*(self.wavelength+self.wfsConfig.wvlBandWidth*1e-9/2.)\
            *self.pxlsPerLP/(self.pxlsPerLP/2.-(self.wavelength+self.wfsConfig.wvlBandWidth*1e-9/2.))))
        if self.nxLP < m:
            logger.warning('nxLP must be at least %d. Using that value'%(m))
            self.nxLP = m+0

        # detectorPxls
        # If pupil_size > detectorPxls, then instead of zooming out
        # on pupil, hence loosing resolution, more pixels are added
        # to an intermediate virtual detector2:

        m = float(self.pupil_size) / (self.nxLP*self.pxlsPerLP)
        if m>1:
            # np.ceil(m) turns it into 2 or 3 so there is a round number
            # of pxlsPerLP2, to make the grating sharp
            self.pxlsPerLP2 = int(self.pxlsPerLP * np.ceil(m))
            print 'pxlsPerLP2 = ',self.pxlsPerLP2
            print 'pupil_size = ',self.pupil_size
        else:
            self.pxlsPerLP2 = int(self.pxlsPerLP * np.ceil(m))

        # detectorPxls2 is for calcFocalPlane. detectorPxls is the true
        # size of the detector in makeDetectorPlane
        self.detectorPxls = self.pxlsPerLP*self.nxLP
        self.detectorPxls2 = self.pxlsPerLP2*self.nxLP

        # Padding for oversampling in fourier space
        self.FFTPadding2 = self.detectorPxls2 * self.wfsConfig.fftOversamp

        # Checkin the extensionWindow is a multiple of pxlsPerLP
        # This is not absolutely necesarry but it is tidier
        if self.extensionWindow%self.pxlsPerLP != 0:
            self.extensionWindow += self.pxlsPerLP-self.extensionWindow%self.pxlsPerLP
            logger.warning('extensionWindow is preferably a multiple of pxlsPerLP. Rounding up to nearest multiple of pxlsPerLP: %d. Using that value'%(self.extensionWindow))

        # If physical prop, must always be at same pixel scale
        # If not, can use less phase points for speed
        ## self.sim_pad or self.simOversize not necesarry in non-physical
        if self.config.propagationMode=="Physical":
            logger.warning('Not ready for non-Geometrical propagation')
            raise Exception('Not ready for non-Geometrical propagation')

        # Getting scaled mask
        self.setMask(self.mask)

        # Not yet in use
        self.referenceImage = self.wfsConfig.referenceImage

        # For polychromatic simulations initial wavefront will be
        # positively or negatively padded to represent other wavelengths
        # before FFT propagation through instrument
        self.FFTPaddingMin = None   # Padding for shorter wvl
        self.FFTPaddingMax = None   # Padding for longer wvl
        temp_nWvl= None             # Temporary number of wvl to simulate
        self.FFTPaddingStep = None  # Padding step to simulate next wvl
        self.FFTPaddingList = None  # Corrected and final number of wvl to simulate
        if self.wfsConfig.wvlBandWidth is not None:
            # nWvl is defined using Rayleigh's criterion between PSF's of
            # different wavelengths. Each wvl will produce a different
            # dispersion for modes different to 0.
            temp_nWvl = np.ceil(self.nxLP*self.wfsConfig.wvlBandWidth*1e-9/(1.22*self.wavelength))
            Nmin = self.FFTPadding2*\
                    (self.wavelength-self.wfsConfig.wvlBandWidth*1e-9/2.)/self.wavelength
            self.FFTPaddingMin = Nmin -Nmin%2 #making even
            Nmax = self.FFTPadding2*\
                    (self.wavelength+self.wfsConfig.wvlBandWidth*1e-9/2.)/self.wavelength
            self.FFTPaddingMax = Nmax -Nmax%2 +2 #making even
            self.FFTPaddingStep = (self.FFTPaddingMax -self.FFTPaddingMin)/temp_nWvl
            ## self.FFTPaddingList values should be multiples of 2
            self.FFTPaddingList = (self.FFTPaddingMin + np.round((np.arange(temp_nWvl)
                    +0.5)*self.FFTPaddingStep/2.)*2).astype(int)
            print 'self.FFTPaddingList: ',self.FFTPaddingList

        ## Allocating memory --------

        # Making Focal Plane Spatial Filter
        # lader and therefor the spatial filter are build always
        # centered around a single pixel. This is so no tilt correction
        # has to be applied to the FFT of the phase.
        # When ifftshift is applied to the spatial filter the center moves
        # to position (0,0) in the array.
        lader = np.arange(self.FFTPadding2) -(self.FFTPadding2
            -self.FFTPadding2%2)/2.
        xgrid,ygrid = np.meshgrid(lader,lader)
        # self.r1 = 1.25*self.wfsConfig.fftOversamp
        self.r1 = 1.25*self.wfsConfig.fftOversamp/4
        self.r2 = self.nxLP*self.wfsConfig.fftOversamp/2.
        h1 = np.sqrt(np.square(xgrid) + np.square(ygrid)) < self.r1
        self.h2 = np.sqrt(np.square(xgrid+2*self.r2)
                + np.square(ygrid)) < self.r2*2/3
        self.fpsf = (h1+self.h2).astype(int)   # focal plane spatial filter

        # Optical planes
        self.wfsDetectorPlane = np.zeros((self.detectorPxls+2*self.extensionWindow,
                                    self.detectorPxls+2*self.extensionWindow),
                                    dtype = DTYPE )
        self.slopes = np.zeros( self.n_measurements )
        self.FPlane = np.zeros((self.FFTPadding2,self.FFTPadding2))
        # FPlaneList has all the simulated wavelengths
        self.FPlaneList = [np.zeros((self.FFTPadding2,
                    self.FFTPadding2),dtype=DTYPE)
                    for i in xrange(self.FFTPaddingList.size)]

        # Distance of carrier (modulating frequency) in FFT of wfsDetectorPlane
        self.carrierDistance = int(float(self.wfsDetectorPlane.shape[0])/self.pxlsPerLP)
        # self.demoEdge = self.wfsDetectorPlane.shape[0]/2 - self.carrierDistance
        # Demodulation mask
        mask_lader = np.arange(self.wfsDetectorPlane.shape[0])-self.wfsDetectorPlane.shape[0]/2
        mask_xgrid,mask_ygrid = np.meshgrid(mask_lader,mask_lader)
        self.demoMask = np.sqrt(np.square(mask_xgrid)+np.square(mask_ygrid)) < self.carrierDistance/3.
        ## END Allocating memory ----


    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction="down")

    def setMask(self, mask):
        super(CAWS, self).setMask(mask)
        try:
            self.mask = FITS.Read('/Users/nsdubost/gitnico/caws/soapy/spiderMask.fits')[1]
            print 'SpiderMask loaded'
        except:
            pass

        # Find the mask to apply to the scaled EField
        coord = int((self.sim_size - self.pupil_size)/2.)
        self.cropMask2 = interp.zoom(self.mask[coord:-coord,coord:-coord],self.detectorPxls2) >= 0.5
        FITS.Write(self.cropMask2,'/Users/nsdubost/gitnico/caws/soapy/cropMask2.fits')

        # Detector Mask
        m = int(float(self.pxlsPerLP2) / self.pxlsPerLP)
        if self.pxlsPerLP < self.pxlsPerLP2:
            self.detectorMask = np.pad(interp.binImgs(self.cropMask2,m)>=m**2/2.,((self.extensionWindow,self.extensionWindow),(self.extensionWindow,self.extensionWindow)),mode='constant')
        else:
            self.detectorMask = np.pad(self.cropMask2,((self.extensionWindow,self.extensionWindow),(self.extensionWindow,self.extensionWindow)),mode='constant')
        # self.detectorMask = aotools.circle(self.detectorPxls/2., self.detectorPxls+self.extensionWindow*2)


        # Making grating = grid
        lader = np.arange(self.detectorPxls2) -(self.detectorPxls2-1)/2.
        xgrid,ygrid = np.meshgrid(lader,lader)
        cropGrid2 = (self.pxlsPerLP2/2.<=(xgrid-self.pxlsPerLP2/4.)
            %self.pxlsPerLP2).astype(int)

        self.cropMaskGrid2 = self.cropMask2*cropGrid2
        FITS.Write(self.cropMaskGrid2,'/Users/nsdubost/gitnico/caws/soapy/cropMaskGrid2.fits')
        self.n_measurements = int(self.detectorMask.sum())


    def initFFTs(self):
        # FFTs will be done multiprocessing for speed
        # self.pool = mp.Pool(mp.cpu_count())
        pass

    def initLGS(self):
        super(CAWS, self).initLGS()
        if self.lgsConfig.uplink:
            lgsObj = getattr(
                    LGS, "LGS_{}".format(self.lgsConfig.propagationMode))
            # This line is wrong. Look shackhartmann.py
            # self.lgs = lgsObj(
            #         self.config, self.soapy_config,
            #         nOutPxls=self.FFTPadding2,
            #         outPxlScale=self.wavelength* np.pi / (180. * 3600)
            #         )/(self.wfsConfig.fftOversamp*self.los.telescope_diameter)

    def allocDataArrays(self):
        """
        Allocate the data arrays the WFS will require

        Determines and allocates the various arrays the WFS will require to
        avoid having to re-alloc memory during the running of the WFS and
        keep it fast.


        """
        self.los.allocDataArrays()


    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel. This is required for WFSs that use centroiding.
        The CAWS doesn't need it.
        """
        pass

    def getStatic(self):
        """
        Computes the static measurements, i.e., slopes with flat wavefront
        """
        self.getEF = True
        self.staticData = None

        # Make flat wavefront, and run through WFS in iMat mode to turn off features
        phs = np.zeros([self.los.n_layers, self.screen_size, self.screen_size]).astype(DTYPE)
        self.staticData = self.frame(
                phs, iMatFrame=True).copy()

        self.getEF = False
        self.slopes = self.frame(
                phs, iMatFrame=True).copy()
#######################################################################


    def zeroData(self, detector=True, FP=True):
        """
        Sets data structures in WFS to zero.

        Parameters:
            detector (bool, optional): Zero the detector? default:True
            FP (bool, optional): Zero intermediate focal plane arrays? default: True
        """

        self.zeroPhaseData()

        if FP:
            for i in range(self.FFTPaddingList.size):
                self.FPlaneList[i][:] = 0

        if detector:
            self.wfsDetectorPlane[:] = 0


    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS.
        For the CAWS, there are multiple focal planes. Only in the context of
        this method focal plane means plane conjugated to the pupil,
        which is where the detector is.
        In all other cases, the focal plane is where the phase gets
        actually focused and the focal plane spatial filter is.

        Parameters:
            intensity (float): The relative intensity of this frame,
            is used when multiple WFS frames taken for extended sources.
        '''
        # CAWS needs to know the phase rather than the EField
        if self.config.propagationMode=="Geometric":
            coord = int((self.sim_size - self.pupil_size)/2.)
            self.cropPhase = (self.los.phase)[coord:-coord, coord:-coord]
            # Have to make phase the correct size if geometric prop
            self.cropPhase = interp.zoom(self.cropPhase,self.detectorPxls2)
            FITS.Write(self.cropPhase,'/Users/nsdubost/gitnico/caws/soapy/cropPhase.fits')

        else:
            logger.warning('Not ready for non-Geometrical propagation')
            raise Exception('Not ready for non-Geometrical propagation')

        ## Multiprocessing ---------------------------
        # pool = mp.Pool(mp.cpu_count())
        # self.FPlaneList = pool.map(self.calcInterferogram,
        #                         range(self.FFTPaddingList.size))
        wfsQueues = []
        wfsProcs = []
        self.FPlaneList = []
        for proc in xrange(self.FFTPaddingList.size):
            wfsQueues.append(Queue())
            wfsProcs.append(Process(target=self.calcInterferogram,
                    args=[proc,self.wavelength,self.FFTPaddingList,self.FFTPadding2,
                    self.FFTPaddingMin,self.FFTPaddingMax,self.detectorPxls2,
                    self.wfsConfig.fftOversamp,self.cropMaskGrid2,self.cropPhase,
                    self.fpsf,self.wfsConfig.wvlBandWidth,wfsQueues[proc]])
                    )
            wfsProcs[proc].daemon = True
            wfsProcs[proc].start()

        for proc in xrange(self.FFTPaddingList.size):
            self.FPlaneList.append(wfsQueues[proc].get())
        ## -------------------------------------------


        self.FPlane[:] = 0
        for i in range(len(self.FPlaneList)):
            self.FPlane += self.FPlaneList[i]

        if intensity !=1:
            self.FPlane *= intensity

        ## Normalise so that integral of energy that made it through
        # the pupil and the grid is unitary
        self.FPlane /= self.cropMaskGrid2.sum()
        FITS.Write(self.FPlane,'/Users/nsdubost/gitnico/caws/soapy/FPlane.fits')


    def calcInterferogram(self,j,wvl,paddingList,basePadding,minP,maxP,Np,
            p,cropMaskGrid2,cropPhase,fpsf,wvlBand,queue):
        '''
        Every wavelength sample has a lowwer and an upper limit.
        The difference between them is the wvlBandWidth of the sample.
        The wvlBandWidth of the sample is used to scale the intensity
        of the sample.
        Applies padding with respect to wavelength, calculates PSF,
        applies focal plane filter re-colimates to get conjugated pupil.
        j               --> which padding in paddingList
        wvl             --> self.wavelength
        paddingList     --> self.FFTPaddingList
        basePadding     --> self.FFTPadding2
        minP            --> self.FFTPaddingMin
        maxP            --> self.FFTPaddingMax
        Np              --> self.detectorPxls2
        p               --> self.wfsConfig.fftOversamp
        cropMaskGrid2   --> self.cropMaskGrid2
        cropPhase       --> self.cropPhase
        fpsf            --> self.fpsf
        wvlBand         --> self.wfsConfig.wvlBandWidth
        '''
        # central wvl of current sample
        wvl_j = wvl * paddingList[j]/basePadding

        wvl_l = 0.                      # lower limit of wavelength sample
        wvl_u = 0.                      # upper limit of wavelength sample
        if j == 0:
            wvl_l = wvl * minP/basePadding
        else:
            wvl_l = (wvl * paddingList[j-1]/basePadding + wvl_j)/2.
        if j == paddingList.size -1:
            wvl_u = wvl * maxP/basePadding
        else:
            wvl_u = (wvl * paddingList[j+1]/basePadding + wvl_j)/2.
        dwvl = wvl_u - wvl_l            # wavelength sample size

        E1 = np.pad(cropMaskGrid2*np.exp(1j*cropPhase*wvl/wvl_j),
        ((Np*(p-1)/2,Np*(p-1)/2),(Np*(p-1)/2,Np*(p-1)/2)),mode='constant')

        if paddingList[j]<basePadding:
            edge = (basePadding-paddingList[j])/2
            E1aux = E1[edge:-edge,edge:-edge]
            E2 = fft2(E1aux)*np.sqrt(dwvl/(wvlBand*1e-9))
            fpaux = ifftshift(fpsf[edge:-edge,edge:-edge])
            FITS.Write(fftshift(np.abs(fpaux*E2)),'/Users/nsdubost/gitnico/caws/soapy/fpsf.fits')
            I3aux = np.abs(ifft2(fpaux*E2))**2
            queue.put(np.pad(I3aux, ((edge,edge),(edge,edge)),
                                            mode='constant'))
        else:
            edge = (paddingList[j]-basePadding)/2
            E1aux = np.pad(E1, ((edge,edge),(edge,edge)) , mode='constant')
            E2 = fft2(E1aux)*np.sqrt(dwvl/(wvlBand*1e-9))
            fpaux = ifftshift(np.pad(fpsf, ((edge,edge),(edge,edge)) , mode='constant'))
            FITS.Write(fftshift(np.abs(fpaux*E2)),'/Users/nsdubost/gitnico/caws/soapy/fpsf.fits')
            I3aux = np.abs(ifft2(fpaux*E2))**2
            if edge!=0:
                queue.put(I3aux[edge:-edge,edge:-edge])
            else:
                queue.put(I3aux)


    def makeDetectorPlane(self):
        '''
        Scales and bins intensity data onto the detector with a given number of
        pixels.

        If required, will first convolve final PSF with LGS PSF, then bin
        PSF down to detector size. Finally puts back into ``wfsFocalPlane``
        array in correct order.

        Note: There is no applyLgsUplink()
        '''
        m = int(float(self.pxlsPerLP2) / self.pxlsPerLP)
        edge = (self.FFTPadding2-(self.detectorPxls2+m*self.extensionWindow*2))/2
        if self.pxlsPerLP < self.pxlsPerLP2:
            self.wfsDetectorPlane = interp.binImgs(self.FPlane[edge:-edge,edge:-edge],m)
        else:
            self.wfsDetectorPlane = self.FPlane[edge:-edge,edge:-edge]


        ## Scale data for correct number of photons
        # zeropoint (float): Photometric zeropoint of mag 0 star in photons/metre^2/seconds
        self.wfsDetectorPlane *= photons_per_mag(
                self.wfsConfig.GSMag, self.cropMaskGrid2, self.los.telescope_diameter/self.detectorPxls2,
                self.wfsConfig.exposureTime, self.soapy_config.sim.photometric_zp
                ) * self.wfsConfig.throughput

        if self.wfsConfig.photonNoise:
            self.addPhotonNoise()

        if self.wfsConfig.eReadNoise!=0:
            self.addReadNoise()

        FITS.Write(self.wfsDetectorPlane,'/Users/nsdubost/gitnico/caws/soapy/wfsDetectorPlane.fits')


    def calculateSlopes(self):
        '''
        Calculates WFS slopes from wfsFocalPlane

        Returns:
            ndarray: array of all WFS measurements
        '''
        FFT = fftshift(fft2(self.wfsDetectorPlane))
        FITS.Write(np.abs(FFT),'/Users/nsdubost/gitnico/caws/soapy/FFTwfsDetectorPlane.fits')
        FFT2 = np.pad(FFT,((0,0),(self.carrierDistance,0)),mode='constant')[:,:self.wfsDetectorPlane.shape[0]]*self.demoMask
        FITS.Write(np.abs(FFT2),'/Users/nsdubost/gitnico/caws/soapy/FFT2wfsDetectorPlane.fits')
        ef = ifft2(ifftshift(FFT2))
        if self.getEF:
            # print 'Getting Statics'
            # print 'self.detectorMask.sum() = ',self.detectorMask.sum()
            self.slopes = ef[self.detectorMask.nonzero()]/np.abs(ef[self.detectorMask.nonzero()])
            # print 'self.slopes.shape = ',self.slopes.shape
        else:
            # print 'Not getting Statics'
            # print 'self.detectorMask.sum() = ',self.detectorMask.sum()
            if np.any(self.staticData):
                slopes = np.angle(ef[self.detectorMask.nonzero()]/self.staticData)
            else:
                slopes = np.angle(ef[self.detectorMask.nonzero()])

            self.slopes = slopes.copy()

        return self.slopes

def photons_per_mag(mag, mask, phase_scale, exposureTime, zeropoint):
    """
    Calculates the number of photons per guide star magnitude

    Parameters:
        mag (int): Magnitude of guide star
        mask (ndarray): 2-d pupil mask. 1 if aperture clear, 0 if not
        phase_scale (float): Size of pupil mask pixel in metres
        exposureTime (float): WFS exposure time in seconds
        zeropoint (float): Photometric zeropoint of mag 0 star in photons/metre^2/seconds

    Returns:
        float: photons per WFS frame
    """
    # ZP of telescope
    n_photons = float(zeropoint) * mask.sum() * phase_scale**2

    # N photons for mag and exposure time
    n_photons *= (10**(-float(mag)/2.5)) * exposureTime

    return n_photons
