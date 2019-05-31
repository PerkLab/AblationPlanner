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
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

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
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

     #INTRO 
    intro = qt.QLabel("Place markups as ablation points onto the segmented lesion \n ")
    parametersFormLayout.addWidget(intro)
    #
    # input volume selector
    #
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
    parametersFormLayout.addRow("Input Volume: ", self.inputVolumeSelector)

    #
    # dose volume selector
    #
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
    parametersFormLayout.addRow("Dose Volume: ", self.doseVolumeSelector)

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
    parametersFormLayout.addRow("Needle tip (markup) list: ", self.markupSelector)

    # connections
    self.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.doseVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)
    space = qt.QLabel("\n")

    #ADD BURNING TIME MANAGEMENT 
    burnTimeSelector = qt.QSpinBox()
    burnTimeSelector.setMinimum(0)
    burnTimeSelector.setMaximum(1800) # 30 mins 
    burnTimeSelector.setSingleStep(10)
    parametersFormLayout.addRow("Burn time (s) at ablation needle tip : ", burnTimeSelector)
    self.burnTimeSelector = burnTimeSelector

    #CALC BUTTON
    calcButton = qt.QPushButton("Calculate Ablation")
    calcButton.toolTip = "Print 'Hello World' in standard output"
    parametersFormLayout.addWidget(calcButton)
    calcButton.connect('clicked(bool)', self.onCalculateAblationClicked)
    self.layout.addStretch(1)
    self.calcButton = calcButton

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
    parametersFormLayout.addRow("Segmentation for DVH calculation : ", self.segmentationSelector)


    #DoseVolumeHistogram BUTTON
    dvhButton = qt.QPushButton("Get Dose Volume Histogram of the current plan")
    parametersFormLayout.addWidget(dvhButton)
    dvhButton.connect('clicked(bool)', self.onGetDVHClicked)
    self.layout.addStretch(2)
    self.dvhButton = dvhButton

    #SET UP NEEDLE PLAN 
    entryText = qt.QLabel("\n Once you are happy with the ablation of the tumor, \nset new fiducials as entry points.\n")
    parametersFormLayout.addWidget(entryText)


    #needleBox1 = qt.QHBoxLayout()
    entryLabel= qt.QLabel("Entry Point = Fiducial ")
    entry = qt.QSpinBox()
    entry.setMinimum(1)
    parametersFormLayout.addRow(entryLabel, entry)
    self.entry = entry

    endLabel = qt.QLabel("Needle Tip = Fiducial ")
    end = qt.QSpinBox()
    end.setMinimum(1)
    parametersFormLayout.addRow(endLabel, end)
    self.end = end


    #ADD NEW NEEDLE BUTTON 
    needleButton = qt.QPushButton("Add new needle at these points")
    needleButton.toolTip = "Print 'Hello World' in standard output"
    parametersFormLayout.addWidget(needleButton)
    needleButton.connect('clicked(bool)', self.onAddNeedleClicked)
    self.layout.addStretch(1)
    self.needleButton = needleButton
    self.needleIndex = 0

    #RESET fIDUCIALS BUTTON
    parametersFormLayout.addRow(space)
    resetFiducialsButton = qt.QPushButton("Reset Fiducials")
    parametersFormLayout.addWidget(resetFiducialsButton)
    resetFiducialsButton.connect('clicked(bool)', self.onResetFiducialsClicked)
    self.layout.addStretch(1)
    self.resetFiducialsButton = resetFiducialsButton

    #DELETE CURRENT NEEDLES BUTTON
    resetNeedleButton = qt.QPushButton("Delete Current Needles")
    parametersFormLayout.addWidget(resetNeedleButton)
    resetNeedleButton.connect('clicked(bool)', self.onDeleteNeedleClicked)
    self.layout.addStretch(2)
    self.resetNeedleButton = resetNeedleButton

    # Create logic
    self.logic = RfAblationLogic()

  def cleanup(self):
    pass

  def onSelect(self):
    pass

  def onCalculateAblationClicked(self):
    inputVolumeNode = self.inputVolumeSelector.currentNode()
    doseVolumeNode = self.doseVolumeSelector.currentNode()
    if inputVolumeNode is None or doseVolumeNode is None:
      logging.error('onCalculateAblationClicked: Invalid anatomic or dose inputImage')
      return
    burnTime = (self.burnTimeSelector.value) #will always have a valid entry of min 0
    if burnTime == 0 : 
    	logging.error('onCalculateAblationClicked: Time set to burn is 0 - no calculation needed')
    result = self.logic.calculateAblationDose(self.inputVolumeSelector.currentNode(), self.doseVolumeSelector.currentNode(), burnTime, self.markupSelector.currentNode())

  def onGetDVHClicked(self):
    doseVolumeNode = self.doseVolumeSelector.currentNode()
    segmentationNode = self.segmentationSelector.currentNode()
    if doseVolumeNode is None or segmentationNode is None : 
      logging.error('onGetDVHClicked: Invalid dose inputImage or invalid segmentation')
      return
    result = self.logic.getDVH(self.doseVolumeSelector.currentNode(), self.segmentationSelector.currentNode())

  def onAddNeedleClicked(self):
    entryFiducial = self.entry.value 
    endFiducial = self.end.value
    markupsNode = self.markupSelector.currentNode()
    self.needleIndex = self.needleIndex + 1 
    if markupsNode is None:
      logging.error('onAddNeedleClicked: Invalid markupsNode selected')
      return

    resultMessage = self.logic.createNeedleModel(entryFiducial, endFiducial, self.inputVolumeSelector.currentNode(), self.markupSelector.currentNode(), self.needleIndex)
    #qt.QMessageBox.information(slicer.util.mainWindow(), 'Slicer Python', resultMessage)
    logging.info(str(resultMessage))

  def onResetFiducialsClicked(self):
    result = self.logic.resetFiducials(self.markupSelector.currentNode())

  def onDeleteNeedleClicked(self):
    result = self.logic.deleteNeedleModels(self.inputVolumeSelector.currentNode())

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
    self.isodoseParameterNode = slicer.vtkMRMLIsodoseNode()
    #self.isodoseParameterNode.SetName("NewIsodose3")
    self.dvhParameterNode = slicer.vtkMRMLDoseVolumeHistogramNode()
    #TODO: Create isodose param. node + DVH param. node

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
    grid = np.divide( np.multiply( np.multiply(gridSize,kx_vec),gridSpacing), (2 * math.pi * 100) ) #TODO : Find why the values are off by /100
    
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
    

    # replace Q propagator with limits for k == 0 
    #   if P == 0, the limit is t 
    #   if P ~= 0, the limit is (1 - exp(-P*t))/P

    '''
    In MATLAB - THERE ARE CURRENTLY NO NAN ELEMENTS IN Q_propagator - will implement later
    if ( numel(P) == 1) && (P == 0)
        Q_propagator(isnan(Q_propagator)) = t
    else
        Q_propagator(isnan(Q_propagator)) = (1 - exp(-P * t)) ./ P
    '''

    T0_propagator = T0_propagator.reshape((120,1))  
    #calculate exact Green's function solution 
    if (len(S) == 1) and ( S == 0 ):
        tempChange = np.real(np.fft.ifftn( np.multiply( T0_propagator, np.fft.fftn(tempMat - blood_ambientTemperature) ) ) )
    else : 
        func = np.multiply( T0_propagator, np.fft.fftn(tempMat - blood_ambientTemperature) ) + np.multiply( Q_propagator, np.fft.fftn(S).reshape((120,1)) )
        tempChange = np.real( np.fft.ifftn( func ) )

    T = tempChange + blood_ambientTemperature
    
    centerPoint = int(math.floor(len(T)/2))

    #Set the doseMap based on the above calculations 
    #only take into account temperature change [ raise ] due to compounding effects of
    # multiple needles 
    doseMap = {}
    tValue = centerPoint
    for i in range(centerPoint+1):
        realTemp = int(round(T[tValue]))
        if realTemp == 37:
            tValue = tValue - 1
            continue
        tempChange = realTemp - 37
        if tempChange <= 2:
            doseMap[i] = 1
        elif tempChange <= 4:
            doseMap[i] = 3
        elif tempChange < 10:
            doseMap[i] = 7
        elif tempChange < 12:
            doseMap[i] = 10
        elif tempChange < 15:
            doseMap[i] = 12
        elif tempChange < 18:
            doseMap[i] = 15
        elif tempChange < 23:
            doseMap[i] = 18
        else :
            doseMap[i] = 23
        tValue = tValue - 1

    return doseMap


  
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
              #multiply dose by 5 to match automatic isodose levels 
              doseVolumeArray[int(pt_IJK[2]), int(pt_IJK[1]), int(pt_IJK[0])] += (doseMap[sphereRadius])
              break


  def calculateAblationDose(self, inputVolumeNode, doseVolumeNode, burnTime, needleTipFiducialNode):
    
    '''
    doseMap = {} #Key = radius Value = dosage
    doseMap[0] = 5
    dose = burnTime
    for rad in range(1,burnTime+1): 
      doseMap[rad] = dose
      dose = dose-1
      #burn time of 5 eg
      #{ 0:5, 1:5, 2:4, 3:3, 4:2, 5:1}
    '''

    doseMap = self.calculateDoseMap(burnTime)
    logging.info('calculated dose map')

    vol = slicer.util.array(doseVolumeNode.GetID())
    vol.fill(0)

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
      self.calculateRadialDoseForFiducial(p_IJK[0:3], doseMap, vol, ijkToRasMatrix, needleTipInRAS, i)

      doseVolumeNode.Modified()

    self.createIsodoseSurfaces(doseVolumeNode)
    #logging.info("Done isodose calculations")

  def createIsodoseSurfaces(self, doseVolumeNode):

    logging.info('calculating Isodose volume')
    #TODO: Have our own isodose parameter node
    #      - Create isodose parameter node in logic constructor. Add it to the scene!
    #      - In this function set the parameters in the parameter node (dose volume node, color table node, using SetAndObserveDoseVolumeNode etc.)
    #      - Get isodose logic: logic = slicer.modules.isodose.logic()
    #      - Calculate isodose: logic.CreateIsodoseSurfaces(self.isodoseParameterNode)

    isodoseLogic = slicer.modules.isodose.logic()
    
    self.isodoseParameterNode = slicer.vtkMRMLIsodoseNode()
    slicer.mrmlScene.AddNode(self.isodoseParameterNode)

    #DOSE VOLUME NODE 
    self.isodoseParameterNode.ShowDoseVolumesOnlyOff()
    self.isodoseParameterNode.ShowScalarBarOn()
    self.isodoseParameterNode.DisableModifiedEventOn()
    self.isodoseParameterNode.SetAndObserveDoseVolumeNode(doseVolumeNode)
    self.isodoseParameterNode.DisableModifiedEventOff()
    #self.isodoseParameterNode.SetAndObserveColorTableNode(isodoseColorTableNode)
    
    #COLOR TABLE NODE
    isodoseColorTableNode = isodoseLogic.SetupColorTableNodeForDoseVolumeNode(doseVolumeNode)
    isodoseLogic.SetNumberOfIsodoseLevels(self.isodoseParameterNode, 8)
    isodoseColorTableNode.SetColor(0, "1" , 0, 1, 0, 0.2)
    isodoseColorTableNode.SetColor(1, "3" ,0.1, 0.9, 0.5, 0.2)
    isodoseColorTableNode.SetColor(2, "7" ,1, 1, 0.4, 0.2)
    isodoseColorTableNode.SetColor(3, "10" , 1, 0.9, 0.1, 0.2)
    isodoseColorTableNode.SetColor(4, "12" ,1, 0.5, 0.1, 0.2)
    isodoseColorTableNode.SetColor(5, "15" ,1, 0, 0, 0.2)
    isodoseColorTableNode.SetColor(6, "18" ,0.6, 0, 0.6, 0.2) #55C [+18] is significant thermal injury
    isodoseColorTableNode.SetColor(7, "23" ,0.4, 0.1, 0, 0.2)

    isodoseLogic.CreateIsodoseSurfaces(self.isodoseParameterNode)
    
    '''
    numOfModelNodesBeforeLoad = len( slicer.util.getNodes('vtkMRMLModelNode*') )
    isodoseWidget = slicer.modules.isodose.widgetRepresentation()

    checkOff = slicer.util.findChildren(widget=isodoseWidget, className='QCheckBox', name='checkBox_ShowDoseVolumesOnly')[0]
    checkOff.checked = False

    #select the doseVolume as the dose volume input
    doseVolumeMrmlNodeCombobox = slicer.util.findChildren(widget=isodoseWidget, className='qMRMLNodeComboBox', name='MRMLNodeComboBox_DoseVolume')[0]
    doseVolumeMrmlNodeCombobox.setCurrentNodeID(doseVolumeNode.GetID())

    #set the number of iso levels
    isoLevelsBox = slicer.util.findChildren(widget=isodoseWidget, className='QSpinBox', name='spinBox_NumberOfLevels')[0]
    isoLevelsBox.setValue(burnTime)
    #Generate Isodose Volume and display
    applyButton = slicer.util.findChildren(widget=isodoseWidget, className='QPushButton', text='Generate isodose')[0]
    applyButton.click()
    '''


  def getDVH(self, doseVolumeNode, segmentationNode):
    #TODO: Have our own DVH parameter node ...
    #slicer.util.selectModule('DoseVolumeHistogram')

    slicer.mrmlScene.AddNode(self.dvhParameterNode)
    self.dvhParameterNode.DisableModifiedEventOn()
    self.dvhParameterNode.SetAndObserveDoseVolumeNode(doseVolumeNode)
    self.dvhParameterNode.DisableModifiedEventOff()
    self.dvhParameterNode.DisableModifiedEventOn()
    self.dvhParameterNode.SetAndObserveSegmentationNode(segmentationNode)
    self.dvhParameterNode.DisableModifiedEventOff()

    dvhLogic = slicer.modules.dosevolumehistogram.logic()
    dvhLogic.ComputeDvh(self.dvhParameterNode)

    #TODO: Get Histogram Plot to automatically show up on 3D space in the screen
    # NOT in the following way 
    dvhWidget = slicer.modules.dosevolumehistogram.widgetRepresentation()
    showHist = slicer.util.findChildren(widget=dvhWidget, className='QPushButton', name="pushButton_ShowAll")[0]
    showHist.click()
    
    '''
    dvhWidget = slicer.modules.dosevolumehistogram.widgetRepresentation()
    segmentsCollapsibleGroupBox = slicer.util.findChildren(widget=dvhWidget, name='CollapsibleGroupBox_Segments')[0]
    segmentsTable = slicer.util.findChildren(widget=dvhWidget, name='SegmentsTableView')[0]
    mrmlNodeComboboxes = slicer.util.findChildren(widget=dvhWidget, className='qMRMLNodeComboBox')
    for mrmlNodeCombobox in mrmlNodeComboboxes:
      if 'vtkMRMLScalarVolumeNode' in mrmlNodeCombobox.nodeTypes:
        doseVolumeNodeCombobox = mrmlNodeCombobox
      elif 'vtkMRMLSegmentationNode' in mrmlNodeCombobox.nodeTypes:
        segmentationNodeCombobox = mrmlNodeCombobox

    showDoseVolumesOnly = slicer.util.findChildren(widget=dvhWidget, className='QCheckBox', name='checkBox_ShowDoseVolumesOnly')[0]
    showDoseVolumesOnly.checked = False #allow us to use a volume and not a dose volume

    doseVolumeNodeCombobox.setCurrentNodeID(doseVolumeNode.GetID()) #get dvh of the volume that contains the dose values
    
    computeDvhButton = slicer.util.findChildren(widget=dvhWidget, text='Compute DVH')[0]
    computeDvhButton.click()

    showHist = slicer.util.findChildren(widget=dvhWidget, className='QPushButton', name="pushButton_ShowAll")[0]
    showHist.click()
    '''
    return True

  def createNeedleModel(self, entryFiducialIndex, endFiducialIndex, inputVolumeNode, needleTipFiducialNode, needleIndex):

    self.needleNodeReferenceRole = 'NeedleRef'
    if inputVolumeNode is None:
      errorMessage = 'Invalid input volume given'
      logging.error('createNeedleModel: ' + errorMessage)
      return errorMessage

    entryPointPosition = [0,0,0]
    needleTipFiducialNode.GetNthFiducialPosition(entryFiducialIndex-1, entryPointPosition)
    endPointPosition = [0,0,0]
    needleTipFiducialNode.GetNthFiducialPosition(endFiducialIndex-1, endPointPosition)
    
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
