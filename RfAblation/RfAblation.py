import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math

#
# RfAblation
#

class RfAblation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "RF Ablation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Radiotherapy"]
    self.parent.dependencies = ["Isodose", "DoseVolumeHistogram"]
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]



#
# RfAblationWidget
#

class RfAblationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Input A plan area 
    #
    inputParametersCollapsibleButton = ctk.ctkCollapsibleButton()
    inputParametersCollapsibleButton.text = "Import Parameters"
    self.layout.addWidget(inputParametersCollapsibleButton)

    #IMPORT NEEDLE PLAN 
    inputFormLayout = qt.QFormLayout(inputParametersCollapsibleButton)
    self.inputParametersSelector = slicer.qMRMLNodeComboBox()
    #self.inputParametersSelector.nodeTypes = 
    self.inputParametersButton = qt.QPushButton('Import these parameters as an ablation plan')
    inputFormLayout.addRow('Select needle plan node : ', self.inputParametersSelector)
    inputFormLayout.addWidget(self.inputParametersButton)

	#
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)
    gridLayout = qt.QGridLayout(parametersCollapsibleButton)

     #INTRO 
    intro = qt.QLabel("Place markups as ablation points onto the segmented lesion \n ")
    gridLayout.addWidget(intro,0,0, 1, 2)
    parameterFormLayout = qt.QFormLayout()
  
    #INPUT VOLUME SELECTOR
    self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
    self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputVolumeSelector.selectNodeUponCreation = True
    self.inputVolumeSelector.addEnabled = False
    self.inputVolumeSelector.removeEnabled = False
    self.inputVolumeSelector.noneEnabled = False
    self.inputVolumeSelector.showHidden = False
    self.inputVolumeSelector.showChildNodeTypes = False
    self.inputVolumeSelector.setMRMLScene( slicer.mrmlScene )
    self.inputVolumeSelector.setToolTip( "Pick the input to the algorithm." )
    parameterFormLayout.addRow('    Input Volume : ', self.inputVolumeSelector)

    #DOSE VOLUME SELECTOR 
    self.doseVolumeSelector = slicer.qMRMLNodeComboBox()
    self.doseVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.doseVolumeSelector.selectNodeUponCreation = True
    self.doseVolumeSelector.addEnabled = True
    self.doseVolumeSelector.removeEnabled = True
    self.doseVolumeSelector.noneEnabled = True
    self.doseVolumeSelector.showHidden = False
    self.doseVolumeSelector.showChildNodeTypes = False
    self.doseVolumeSelector.setMRMLScene( slicer.mrmlScene )
    self.doseVolumeSelector.setToolTip( "Pick the output to the algorithm." )
    parameterFormLayout.addRow('    Dose Volume : ', self.doseVolumeSelector)

    # connections
    self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.doseVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    #SEGMENTATION NODE SELECTOR 
    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.segmentationSelector.selectNodeUponCreation = True
    self.segmentationSelector.addEnabled = True
    self.segmentationSelector.removeEnabled = True
    self.segmentationSelector.removeEnabled = True
    self.segmentationSelector.noneEnabled = True
    self.segmentationSelector.showHidden = False
    self.segmentationSelector.showChildNodeTypes = False
    self.segmentationSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationSelector.setToolTip("Pick the correct segmentation node")
    self.segmentationSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.populateLesionSegmentSelection )
    parameterFormLayout.addRow('    Segmentation Node : ', self.segmentationSelector)

    #Set Lesion segment - This combobox will be populated based on the selected segmentation node
    self.lesionSelector = qt.QComboBox()
    parameterFormLayout.addRow('   Lesion Segment : ', self.lesionSelector)

    gridLayout.addLayout(parameterFormLayout,1,0)

    #Margin Button 
    marginButton = qt.QPushButton("Apply")
    marginButton.connect('clicked(bool)', self.onCalcMarginClicked)
    self.marginButton = marginButton
    marginSizeSelector = qt.QSpinBox()
    marginSizeSelector.setMinimum(0)
    marginSizeSelector.setSuffix(' mm')
    self.marginSizeSelector = marginSizeSelector

    marginFormLayout = qt.QFormLayout()
    marginFormLayout.addRow("   Set a margin of : ", self.marginSizeSelector)
    gridLayout.addLayout(marginFormLayout,2,0)
    gridLayout.addWidget(self.marginButton, 2, 1)

    # MarkUp list selector
    #
    self.markupSelector = slicer.qMRMLNodeComboBox()
    self.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.markupSelector.selectNodeUponCreation = True
    self.markupSelector.addEnabled = True
    self.markupSelector.removeEnabled = True
    self.markupSelector.noneEnabled = True
    self.markupSelector.showHidden = False
    self.markupSelector.showChildNodeTypes = False
    self.markupSelector.setMRMLScene( slicer.mrmlScene )
    self.markupSelector.setToolTip( "Pick the markup list" )
    self.markupSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onMarkupsNodeSelectionChanged)

    #ADD BURNING TIME MANAGEMENT 
    burnTimeSelector = qt.QSpinBox()
    burnTimeSelector.setMinimum(0)
    burnTimeSelector.setMaximum(1800) # 30 mins 
    burnTimeSelector.setValue(200) #recommended minimum 
    burnTimeSelector.setSingleStep(10)
    burnTimeSelector.setSuffix('s')
    self.burnTimeSelector = burnTimeSelector

    markupFormLayout = qt.QFormLayout()
    markupFormLayout.addRow('   Pick the markup list: ', self.markupSelector)
    markupFormLayout.addRow('   Set burn time :', self.burnTimeSelector)
    gridLayout.addLayout(markupFormLayout, 3,0)

    #CALC BUTTON
    calcButton = qt.QPushButton("Calculate Ablation")
    calcButton.toolTip = "Print 'Hello World' in standard output"
    calcButton.connect('clicked(bool)', self.onCalculateAblationClicked)
    self.layout.addStretch(1)
    self.calcButton = calcButton

    gridLayout.addWidget(calcButton, 4,0)

    #DoseVolumeHistogram BUTTON
    dvhButton = qt.QPushButton("Get Dose Volume Histogram of the current plan")
    dvhButton.connect('clicked(bool)', self.onGetDVHClicked)
    self.layout.addStretch(2)
    self.dvhButton = dvhButton

    gridLayout.addWidget(dvhButton,5,0)

    #
    # Set Needle Plan 
    #
    needleParametersCollapsibleButton = ctk.ctkCollapsibleButton()
    needleParametersCollapsibleButton.text = "Set Needle Plan"
    self.layout.addWidget(needleParametersCollapsibleButton)

    #IMPORT NEEDLE PLAN 
    needleFormLayout = qt.QFormLayout(needleParametersCollapsibleButton)

    #SET UP NEEDLE PLAN 
    entryText = qt.QLabel("\n Once you are happy with the ablation of the tumor, set new fiducials as entry points.\n")
    needleFormLayout.addWidget(entryText)

    #ENTRY POINT 
    self.needleEntryCombobox = qt.QComboBox()
    self.needleEntryCombobox.connect('activated(int)',self.populateNeedleEntryComboBox)
    #NEEDLE TIP POINT 
    self.needleTipCombobox = qt.QComboBox()

    #ADD NEW NEEDLE BUTTON 
    needleButton = qt.QPushButton("Add new needle at these points")
    needleButton.toolTip = "Print 'Hello World' in standard output"
    needleButton.connect('clicked(bool)', self.onAddNeedleClicked)
    self.layout.addStretch(1)
    self.needleButton = needleButton
    self.needleIndex = 0

    self.needleFiducialPairList = []

    needleFormLayout.addRow('   Needle Entry point : ', self.needleEntryCombobox)
    needleFormLayout.addRow('   Needle Tip point : ', self.needleTipCombobox)
    needleFormLayout.addWidget(needleButton)


    #RESET fIDUCIALS BUTTON
    resetFiducialsButton = qt.QPushButton("Reset Fiducials")
    resetFiducialsButton.connect('clicked(bool)', self.onResetFiducialsClicked)
    self.resetFiducialsButton = resetFiducialsButton
    needleFormLayout.addWidget(self.resetFiducialsButton)

    #DELETE CURRENT NEEDLES BUTTON
    resetNeedleButton = qt.QPushButton("Delete Current Needles")
    resetNeedleButton.connect('clicked(bool)', self.onDeleteNeedleClicked)
    self.resetNeedleButton = resetNeedleButton
    needleFormLayout.addWidget(self.resetNeedleButton)

    # Create logic
    self.logic = RfAblationLogic()

    #Create or get parameter node 
    self.rfaParameterNode = self.setParameters()


  def setParameters(self) :

    #
    #Load parameter node or create a new one if one does not exist
    #
    #
    scriptedModuleLogic = slicer.ScriptedLoadableModule.ScriptedLoadableModuleLogic()
    self.rfaParameterNode = scriptedModuleLogic.getParameterNode()
    paramNum = self.rfaParameterNode.GetParameterCount()

    print paramNum, "Number of parameters"

    #Set parameter name constants
    self.INPUT_VOLUME = "inputVolumeName"
    self.DOSE_VOLUME = "doseVolumeName"
    self.SEGMENTATION = "segmentationName"
    self.MARGIN_SIZE_MM = "marginSizeMm"
    self.MARKUP_LIST = "markupNodeName"
    self.BURN_TIME = "burnTime"

    #Check if the loaded scene was created with any of the following parameters 
    #If so load them in
    if paramNum != 0 :
    	setParameters = self.rfaParameterNode.GetParameterNames()
    	for name in setParameters:
    		value = self.rfaParameterNode.GetParameter(name)
    		if name == self.INPUT_VOLUME:
    			node = slicer.mrmlScene.GetNodeByID(value)
    			if node is None:
    				logging.warning('referenced input volume does not exist - select new volume')
    			else:
    				self.inputVolumeSelector.setCurrentNode(node)
    		if name == self.DOSE_VOLUME:
    			node = slicer.mrmlScene.GetNodeByID(value)
    			if node is None:
    				logging.warning('referenced dose volume does not exist - select new volume')
    			else :
    				self.doseVolumeSelector.setCurrentNode(node)
    		if name == self.SEGMENTATION:
    			node = slicer.mrmlScene.GetNodeByID(value)
    			if node is None:
    				logging.warning('referenced segmentation node does not exist - select new node')
    			else :
    				self.segmentationSelector.setCurrentNode(node)
    				self.populateLesionSegmentSelection()
    		if name == self.MARGIN_SIZE_MM:
    			self.marginSizeSelector.setValue(int(value))
    		if name == self.MARKUP_LIST:
    			node = slicer.mrmlScene.GetNodeByID(value)
    			if node is None:
    				logging.warning('referenced markup list does not exist - select new markup node')
    			else :
    				self.markupSelector.setCurrentNode(node)
    		if name == self.BURN_TIME:
    			self.burnTimeSelector.setValue(int(value))

    return self.rfaParameterNode
  
    

  def cleanup(self):
    pass

  def onSelect(self):
    pass

  def onCalcMarginClicked(self):
    inputVolumeNode = self.inputVolumeSelector.currentNode()
    segmentationNode = self.segmentationSelector.currentNode()
    if inputVolumeNode is None or segmentationNode is None : 
        logging.error('onCalcMarginClicked: Invalid input volume or segmentation node')
        return
    lesionSegment = self.lesionSelector.currentText
    print lesionSegment
    if lesionSegment is None:
    	loggin.error('onCalcMarginClicked: Invalid or no lesion segment selected')
    	return
    marginSizeMm = self.marginSizeSelector.value #minimum set to 0 in selector 
    result = self.logic.applyTumourMargin(self.inputVolumeSelector.currentNode(), self.segmentationSelector.currentNode(), self.lesionSelector.currentText,marginSizeMm)

    self.rfaParameterNode.SetParameter(self.SEGMENTATION, segmentationNode.GetID() )
    self.rfaParameterNode.SetParameter(self.MARGIN_SIZE_MM, str(marginSizeMm) )

  def onCalculateAblationClicked(self):
    inputVolumeNode = self.inputVolumeSelector.currentNode()
    doseVolumeNode = self.doseVolumeSelector.currentNode()
    markupNode = self.markupSelector.currentNode()
    if inputVolumeNode is None or doseVolumeNode is None :
      logging.error('onCalculateAblationClicked: Invalid anatomic or dose inputImage')
      return
    if doseVolumeNode.GetImageData() is None:
    	#User has not imported a dose volume and is creating a new volume
    	volumeLogic = slicer.modules.volumes.logic()
    	doseVolumeName = 'DoseVolume' + str(inputVolumeNode.GetName())
    	doseVolumeNode = volumeLogic.CloneVolume(inputVolumeNode, doseVolumeName)
    if markupNode is None :
    	logging.error('onCalculateAblationClicked: Invalid markup node')
    	return 
    burnTime = (self.burnTimeSelector.value) #will always have a valid entry of min 0
    if burnTime == 0 : 
    	logging.error('onCalculateAblationClicked: Time set to burn is 0 - no calculation needed')
    result = self.logic.calculateAblationDose(self.inputVolumeSelector.currentNode(), doseVolumeNode, burnTime, self.markupSelector.currentNode())

    self.rfaParameterNode.SetParameter(self.INPUT_VOLUME, inputVolumeNode.GetID() )
    self.rfaParameterNode.SetParameter(self.DOSE_VOLUME, doseVolumeNode.GetID() )
    self.rfaParameterNode.SetParameter(self.MARKUP_LIST, markupNode.GetID() )
    self.rfaParameterNode.SetParameter(self.BURN_TIME, str(burnTime))
   	#TODO : USE A REFERENCE ID FOR THE VOLUMES 

  def populateLesionSegmentSelection(self):
  	segmentationNode = self.segmentationSelector.currentNode()
  	if segmentationNode is None: 
  		logging.error('populateLesionSegmentSelection: no segmentation node')
  		return
  	self.lesionSelector.clear()
  	segmentation = segmentationNode.GetSegmentation()
  	numberOfSegments = segmentation.GetNumberOfSegments()
  	for segment in range(numberOfSegments):
  		segmentName = segmentation.GetNthSegment(segment).GetName()
  		self.lesionSelector.addItem(segmentName, segment)

  def onMarkupsNodeSelectionChanged(self):
    self.markupsNode = self.markupSelector.currentNode()
    if self.markupsNode is None:
        logging.error('onMarkupsNodeSelectionChanged: Invalid Markups Node')
        return

    self.markupsNodeObserver = self.markupsNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateFiducialModels )
    self.oldnumberOfMarkers = 0
    self.populateNeedleEntryComboBox()

  def updateFiducialModels(self, caller=None, event=None):
  	#self.needleNodeReferenceRole = 'NeedleRef'
  	inputVolumeNode = self.inputVolumeSelector.currentNode()
  	print "fiducial being moved"
  	numberOfNeedles = len(self.needleFiducialPairList)
  	if numberOfNeedles > 0:
  		print numberOfNeedles
  		for num in range(numberOfNeedles):
  			needleInfo = self.needleFiducialPairList[num]
  			needleModelNode = inputVolumeNode.GetNthNodeReference('NeedleRef', (needleInfo[2]-1)) #TODO: remove the hard coded name
			slicer.mrmlScene.RemoveNode(needleModelNode)

		for create in range(numberOfNeedles):
			needleInfo = self.needleFiducialPairList[create]
			print create, needleInfo
  			result = self.logic.createNeedleModel(needleInfo[0],needleInfo[1], inputVolumeNode, self.markupSelector.currentNode(), needleInfo[2])


  def populateNeedleEntryComboBox(self):
    markupsNode = self.markupSelector.currentNode()
    numberOfMarkers = markupsNode.GetNumberOfFiducials()
    if (numberOfMarkers != self.oldnumberOfMarkers) == True :
        self.oldnumberOfMarkers = numberOfMarkers
        self.needleEntryCombobox.clear()
        self.needleTipCombobox.clear()
        if numberOfMarkers > 0 :
            self.needleEntryCombobox.enabled = True 
        else :
            logging.error('populateNeedleEntryComboBox: No fiducials to choose from ')
            return 

        for markupIndex in range(numberOfMarkers):
            label = markupsNode.GetNthFiducialLabel(markupIndex)
            self.needleEntryCombobox.addItem(label, markupIndex)
            self.needleTipCombobox.addItem(label, markupIndex)


  def onGetDVHClicked(self):
    doseVolumeNode = self.doseVolumeSelector.currentNode()
    segmentationNode = self.segmentationSelector.currentNode()
    if doseVolumeNode is None or segmentationNode is None : 
      logging.error('onGetDVHClicked: Invalid dose inputImage or invalid segmentation')
      return
    result = self.logic.getDVH(self.doseVolumeSelector.currentNode(), self.segmentationSelector.currentNode())

  def onAddNeedleClicked(self):
    markupsNode = self.markupSelector.currentNode()
    entryPointIndex = self.needleEntryCombobox.currentIndex
    tipPointIndex = self.needleTipCombobox.currentIndex
    self.needleIndex = self.needleIndex + 1 # increase index at every new needle added 
    if markupsNode is None:
      logging.error('onAddNeedleClicked: Invalid markupsNode selected')
      return

    self.needleFiducialPairList.append([entryPointIndex, tipPointIndex, self.needleIndex])

    resultMessage = self.logic.createNeedleModel(entryPointIndex, tipPointIndex, self.inputVolumeSelector.currentNode(), self.markupSelector.currentNode(), self.needleIndex)
    logging.info(str(resultMessage))

  def onResetFiducialsClicked(self):
    result = self.logic.resetFiducials(self.markupSelector.currentNode())
    self.needleEntryCombobox.clear()
    self.needleTipCombobox.clear()

  def onDeleteNeedleClicked(self):
    result = self.logic.deleteNeedleModels(self.inputVolumeSelector.currentNode())
    self.needleIndex = 0
    self.needleFiducialPairList = [] 
#
# RfAblationLogic
#

class RfAblationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):

    self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(self.segmentEditorNode)

    self.isodoseParameterNode = slicer.vtkMRMLIsodoseNode()
    slicer.mrmlScene.AddNode(self.isodoseParameterNode)

    self.dvhParameterNode = slicer.vtkMRMLDoseVolumeHistogramNode()
    slicer.mrmlScene.AddNode(self.dvhParameterNode)

    
  def applyTumourMargin(self, inputVolumeNode, segmentationNode, lesionSegment, marginSizeMm) :

    self.segmentEditorNode.SetAndObserveMasterVolumeNode(inputVolumeNode)
    self.segmentEditorNode.SetAndObserveSegmentationNode(segmentationNode)

    import vtkSegmentationCorePython as vtkSegmentationCore
    marginSegmentID = 'marginID'
    marginSegmentName = segmentationNode.GetName() + '_margin'
    marginColor = 80 
    marginSegment = segmentationNode.GetSegmentation().AddEmptySegment(marginSegmentID, marginSegmentName, [0.1,0.5,1])

    modifierID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(lesionSegment) #TODO: INPUT 
    modifierSegment = segmentationNode.GetSegmentation().GetSegment(modifierID)
    modifierSegmentLabelmap = modifierSegment.GetRepresentation(vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())

    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(modifierSegmentLabelmap,
            segmentationNode, marginSegmentID, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, modifierSegmentLabelmap.GetExtent())

    labelMapSpacing = modifierSegmentLabelmap.GetSpacing()
    kernelSizePixel = [int(round((marginSizeMm / labelMapSpacing[componentIndex]+1)/2)*2-1) for componentIndex in range(3)]
    marginSegment = segmentationNode.GetSegmentation().GetSegment(marginSegmentID)
    selectedSegmentLabelmap = marginSegment.GetRepresentation(vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())

    # We need to know exactly the value of the segment voxels, apply threshold to make force the selected label value
    labelValue = 1
    backgroundValue = 0
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(selectedSegmentLabelmap)
    thresh.ThresholdByLower(0)
    thresh.SetInValue(backgroundValue)
    thresh.SetOutValue(labelValue)
    thresh.SetOutputScalarType(selectedSegmentLabelmap.GetScalarType())

    erodeDilate = vtk.vtkImageDilateErode3D()
    erodeDilate.SetInputConnection(thresh.GetOutputPort())
    #grow 
    erodeDilate.SetDilateValue(labelValue)
    erodeDilate.SetErodeValue(backgroundValue)
    erodeDilate.SetKernelSize(kernelSizePixel[0],kernelSizePixel[1],kernelSizePixel[2])
    erodeDilate.Update()
    selectedSegmentLabelmap.DeepCopy(erodeDilate.GetOutput())

    selectedSegmentLabelmap.Modified()
    create = segmentationNode.CreateClosedSurfaceRepresentation()


  def makeDim(self, gridSize, gridSpacing):
    #define the discretisation of the spatial dimension such that
    # there is always a DC component
    if (gridSize % 2) == 0:
        #grid dimension has an even number of points
        #nx = ((-Nx/2:Nx/2-1)/Nx).'
        dim = np.arange(-gridSize/2,gridSize/2,1)
        nx = np.true_divide(dim,gridSize)
    else :
        # grid dimension has an odd number of points
        #nx = ((-(Nx-1)/2:(Nx-1)/2)/Nx).'
        dim = np.arange(-(gridSize-1)/2,(gridSize-1)/(2+1),1)
        nx = np.true_divide(dim,gridSize)

    # force middle value to be zero in case 1/Nx is a recurring
    # number and the series doesn't give exactly zero
    nx[int(math.floor(gridSize/2))] = 0
            
    # define the wavenumber vector components
    kx_vec = np.multiply((2*math.pi/gridSpacing), nx);

    return kx_vec # CHECKED 

  def calculateDoseMap(self, burnTime):

    temperatureProfile = self.calculateTemperatureProfile(burnTime)

    centerPoint = int(math.floor(len(temperatureProfile)/2))

    #Set the doseMap based on the above calculations 
    #only take into account temperature change [ raise ] due to compounding effects of
    # multiple needles 
    doseMap = {}
    tempIndex = centerPoint
    print temperatureProfile[centerPoint]
    for i in range(centerPoint+1):
        realTemp = int(round(temperatureProfile[tempIndex]))
        if realTemp == 37:
            tempIndex = tempIndex - 1
            continue
        tempChange = realTemp - 37
        if tempChange <= 2:
            doseMap[i] = 39
        elif tempChange <= 4:
            doseMap[i] = 41
        elif tempChange < 10:
            doseMap[i] = 46
        elif tempChange < 12:
            doseMap[i] = 48
        elif tempChange < 15:
            doseMap[i] = 51
        elif tempChange < 18:
            doseMap[i] = 54
        elif tempChange < 23:
            doseMap[i] = 59
        elif tempChange < 43:
            doseMap[i] = 65
        else :
        	doseMap[i] = 100
        tempIndex = tempIndex - 1

    return doseMap

  def calculateTemperatureProfile(self, burnTime):
    #LIVER TISSUE PROPERTIES 
    density = 1079 # [kg/m^3]
    thermalConductivity = 0.52 # [W/(m.K) ]
    specificHeat = 3540 # [J/(kg.K)]
    ambientTemperature = 37 # [degC]

    blood_density = 1060 
    blood_specificHeat = 3617
    blood_perfusionRate = 0.01 #[1/s]
    blood_ambientTemperature = 37 #[degC]

    #Set time at which to calculate temperature field 
    t = burnTime

    #calculate perfusion coefficient from the medium 
    P = (blood_density * blood_perfusionRate * blood_specificHeat) / (density * specificHeat)

    #calculate diffusivity from the medium 
    D = thermalConductivity / (density * specificHeat)

    gridSpacing = 0.01 #distance between points in the grid [m] 
    gridSize = 120 
    kx_vec = self.makeDim(gridSize, gridSpacing) #CHECKED : same output as matlab 

    #kgrid.x - grid containing repeated copies of the grid coordinates in the x-direction 
    grid = np.divide( np.multiply( np.multiply(gridSize,kx_vec),gridSpacing), (2 * math.pi * 100) ) 
    
    
    #kgrid.k - Nx x Ny x Nz grid of the scalar wavenumber where k = sqrt(kx.^2 + ky.^2 + kz.^2) [rad/m]
    kgrid = np.sqrt(np.power( kx_vec,2 ) )

    width = 4 * gridSpacing
    #calculate volume rate of heat deposition 
        #set Gaussian volume rate of heat deposition 
    volumeRate = np.multiply(2000000, np.exp( -np.power((np.divide(grid,width)),2) ) ) 

    S = np.divide(volumeRate ,np.multiply(density,specificHeat)) #CHECKED 

    #Initialize matrix of initial temperatures that is the same size as S 
    tempMat = np.full((gridSize,1),ambientTemperature)

    # Define Green's funciton propagators 
    #ifftshift gives different results in matlab than it does in Python 
    ftShift = np.power( np.fft.ifftshift(kgrid) , 2 )

    T0_propagator = np.exp( np.multiply( -( np.multiply(D,ftShift) + P),t) ) #CHECKED
    Q_propagator = np.divide( (1 - T0_propagator), ( np.multiply(D,ftShift) + P ) ).reshape((120,1))   #CHECKED 
    T0_propagator = T0_propagator.reshape((120,1))  
    #calculate exact Green's function solution 
    if (len(S) == 1) and ( S == 0 ):
        tempChange = np.real(np.fft.ifftn( np.multiply( T0_propagator, np.fft.fftn(tempMat - blood_ambientTemperature) ) ) )
    else : 
        func = np.multiply( T0_propagator, np.fft.fftn(tempMat - blood_ambientTemperature) ) + np.multiply( Q_propagator, np.fft.fftn(S).reshape((120,1)) )
        tempChange = np.real( np.fft.ifftn( func ) )

    T = tempChange + blood_ambientTemperature
     
    return T

  
  def calculateRadialDoseForFiducial(self, needleTip, doseMap, doseVolumeArray, ijkToRasMatrix, needleTipInRAS, needleTipIndex):

    logging.info('Calculating Radial Dose for Needle Tip ' + str(needleTipIndex))
    rad = len(doseMap)-1 #radius of 0 is included in the doseMap 
    #search through cube of the radius and select spherical points
    for i in range(-rad, rad + 1):
      for j in range(-rad, rad + 1):
        for k in range(-rad, rad + 1):
          pos = np.array([i, j, k])
          pt_IJK = pos + np.array(needleTip)  # point in ijk
          appended_pt = np.append(pt_IJK,1)
          pt_RAS = ijkToRasMatrix.MultiplyDoublePoint(appended_pt)
          euclDist = np.linalg.norm(np.array(pt_RAS[0:3]) - np.array(needleTipInRAS))
          for sphereRadius in range(len(doseMap)):
            if euclDist <= sphereRadius:
              doseVolumeArray[int(pt_IJK[2]), int(pt_IJK[1]), int(pt_IJK[0])] += (doseMap[sphereRadius])
              break


  def calculateAblationDose(self, inputVolumeNode, doseVolumeNode, burnTime, needleTipFiducialNode):

    doseMap = self.calculateDoseMap(burnTime)
    logging.info('calculated dose map')
    print doseMap

    doseVolumeArray = slicer.util.array(doseVolumeNode.GetID())
    doseVolumeArray.fill(0)

    num = needleTipFiducialNode.GetNumberOfFiducials()

    rasToIjkMatrix = vtk.vtkMatrix4x4()
    inputVolumeNode.GetRASToIJKMatrix(rasToIjkMatrix)
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

    for i in range(num):
      pos = [0, 0, 0]
      needleTipFiducialNode.GetNthFiducialPosition(i, pos)
      needleTipInRAS=pos
      pos = np.append(pos, 1)

      p_IJK = rasToIjkMatrix.MultiplyDoublePoint(pos)
      self.calculateRadialDoseForFiducial(p_IJK[0:3], doseMap, doseVolumeArray, ijkToRasMatrix, needleTipInRAS, i)

      doseVolumeNode.Modified()

    self.createIsodoseSurfaces(doseVolumeNode)
    #logging.info("Done isodose calculations")

  def createIsodoseSurfaces(self, doseVolumeNode):

    logging.info('calculating Isodose volume')
    isodoseLogic = slicer.modules.isodose.logic()
    
    #DOSE VOLUME NODE 
    self.isodoseParameterNode.ShowDoseVolumesOnlyOff()
    self.isodoseParameterNode.ShowScalarBarOn()
    self.isodoseParameterNode.DisableModifiedEventOn()
    self.isodoseParameterNode.SetAndObserveDoseVolumeNode(doseVolumeNode)
    self.isodoseParameterNode.DisableModifiedEventOff()
    
    #COLOR TABLE NODE
    isodoseColorTableNode = isodoseLogic.SetupColorTableNodeForDoseVolumeNode(doseVolumeNode)
    isodoseLogic.SetNumberOfIsodoseLevels(self.isodoseParameterNode, 8)
    isodoseColorTableNode.SetColor(0, "39" , 0, 1, 0, 0.2)
    isodoseColorTableNode.SetColor(1, "41" ,0.1, 0.9, 0.5, 0.2)
    isodoseColorTableNode.SetColor(2, "46" ,1, 1, 0.4, 0.2)
    isodoseColorTableNode.SetColor(3, "48" , 1, 0.9, 0.1, 0.2)
    isodoseColorTableNode.SetColor(4, "51" ,1, 0.5, 0.1, 0.2)
    isodoseColorTableNode.SetColor(5, "54" ,1, 0, 0, 0.2)
    isodoseColorTableNode.SetColor(6, "59" ,0.6, 0, 0.6, 0.2) #55C [+18] is significant thermal injury
    isodoseColorTableNode.SetColor(7, "65" ,0.4, 0.1, 0, 0.2)

    isodoseLogic.CreateIsodoseSurfaces(self.isodoseParameterNode)


  def getDVH(self, doseVolumeNode, segmentationNode):

    self.dvhParameterNode.DisableModifiedEventOn()
    self.dvhParameterNode.SetAndObserveDoseVolumeNode(doseVolumeNode)
    self.dvhParameterNode.DisableModifiedEventOff()
    self.dvhParameterNode.DisableModifiedEventOn()
    self.dvhParameterNode.SetAndObserveSegmentationNode(segmentationNode)
    self.dvhParameterNode.DisableModifiedEventOff()

    dvhLogic = slicer.modules.dosevolumehistogram.logic()
    dvhLogic.ComputeDvh(self.dvhParameterNode)

    #Show Intensity volume Histogram 
    metricsTableNode = self.dvhParameterNode.GetMetricsTableNode()
    visibilityColumn = metricsTableNode.GetTable().GetColumn(self.dvhParameterNode.MetricColumnVisible)

    numOfRows = metricsTableNode.GetNumberOfRows()
    for row in range(numOfRows):
        visibilityColumn.SetValue(row, True)

    visibilityColumn.Modified()
    metricsTableNode.Modified()

    return True

  def createNeedleModel(self, entryFiducialIndex, endFiducialIndex, inputVolumeNode, needleTipFiducialNode, needleIndex):

    self.needleNodeReferenceRole = 'NeedleRef'
    if inputVolumeNode is None:
      errorMessage = 'Invalid input volume given'
      logging.error('createNeedleModel: ' + errorMessage)
      return errorMessage

    entryPointPosition = [0,0,0]
    needleTipFiducialNode.GetNthFiducialPosition(entryFiducialIndex, entryPointPosition)
    endPointPosition = [0,0,0]
    needleTipFiducialNode.GetNthFiducialPosition(endFiducialIndex, endPointPosition)
    
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(entryPointPosition[0],entryPointPosition[1],entryPointPosition[2])
    lineSource.SetPoint2(endPointPosition[0],endPointPosition[1],endPointPosition[2])
    lineSource.Update()
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputConnection(lineSource.GetOutputPort())
    tubeFilter.SetRadius(0.5) #Default is 0.5
    tubeFilter.SetNumberOfSides(100)
    tubeFilter.Update()

    needleModelNode = slicer.vtkMRMLModelNode()
    needleModelName = 'Needle_' + inputVolumeNode.GetName() + '_' + str(needleIndex)
    needleModelNode.SetName(needleModelName)
    slicer.mrmlScene.AddNode(needleModelNode)
    needleModelNode.SetPolyDataConnection(tubeFilter.GetOutputPort())
    inputVolumeNode.AddNodeReferenceID(self.needleNodeReferenceRole, needleModelNode.GetID())

    modelDisplayNode = slicer.vtkMRMLModelDisplayNode()
    modelDisplayNode.SetColor(1,1,0)
    modelDisplayNode.SetOpacity(1)
    modelDisplayNode.SetSliceIntersectionThickness(2)
    modelDisplayNode.SetSliceIntersectionVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplayNode)
    needleModelNode.SetAndObserveDisplayNodeID(modelDisplayNode.GetID())
    return "Needle Successfully added"


  def resetFiducials(self, needleTipFiducialNode ):
    needleTipFiducialNode.RemoveAllMarkups()
    return True


  def deleteNeedleModels(self, inputVolumeNode):
    self.needleNodeReferenceRole = 'NeedleRef'

    if inputVolumeNode is None:
      logging.error('resetNeedlePlan: Invalid input volume')
      return False

    # Needle models are automatically created with the prefix of Model
    numberOfNeedles = inputVolumeNode.GetNumberOfNodeReferences(self.needleNodeReferenceRole)
    needleModelNodesToRemove = []
    for refIndex in range(numberOfNeedles):
      needleModelNode = inputVolumeNode.GetNthNodeReference(self.needleNodeReferenceRole, refIndex)
      if needleModelNode is not None:
        needleModelNodesToRemove.append(needleModelNode)

    # 
    for needleModelNode in needleModelNodesToRemove:
      slicer.mrmlScene.RemoveNode(needleModelNode)

    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True



class RfAblationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_RfAblation1()

  def test_RfAblation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = RfAblationLogic()
    #TODO: Later
