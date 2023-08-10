import vtk
import random
import numpy as np
import time
from vtk.util import numpy_support as ns
from scipy.interpolate import griddata        
import math


# Function to randomly sample dataset points. (returns sampled points coordinates, sampled points and value array) 

def SRS_function(input_data,sampling_per):
    dims = input_data.GetDimensions()                           
    num_points = input_data.GetNumberOfPoints()    
    num_samples = int(num_points * sampling_per / 100)  # Compute the number of points to sample

    sampled_points = vtk.vtkPoints()
    Data_Value_Array = ns.vtk_to_numpy(input_data.GetPointData().GetArray('Pressure'))   # Storing all point values in a numpy array (Data_Value_Array)
    sampled_data_value = []

    # Create an array to store the coordinates for each sampled point
    tup_list = []
    val_scalar = vtk.vtkFloatArray()
    # Select the eight corner points of the volume

    sampled_points.InsertNextPoint(input_data.GetPoint(0))
    tup_list.append(input_data.GetPoint(0))
    sampled_data_value.append(Data_Value_Array[0])
    val_scalar.InsertNextValue(Data_Value_Array[0])

    sampled_points.InsertNextPoint(input_data.GetPoint(dims[0]-1))
    tup_list.append(input_data.GetPoint(dims[0]-1))
    sampled_data_value.append(Data_Value_Array[dims[0]-1])
    val_scalar.InsertNextValue(Data_Value_Array[dims[0]-1])


    sampled_points.InsertNextPoint(input_data.GetPoint((dims[0]*dims[1])-dims[0]))
    tup_list.append(input_data.GetPoint((dims[0]*dims[1])-dims[0]))
    sampled_data_value.append(Data_Value_Array[((dims[0]*dims[1])-dims[0])])
    val_scalar.InsertNextValue(Data_Value_Array[((dims[0]*dims[1])-dims[0])])

    sampled_points.InsertNextPoint(input_data.GetPoint((dims[0]*dims[1])-1))
    tup_list.append(input_data.GetPoint((dims[0]*dims[1])-1))
    sampled_data_value.append(Data_Value_Array[(dims[0]*dims[1])-1])
    val_scalar.InsertNextValue(Data_Value_Array[(dims[0]*dims[1])-1])


    sampled_points.InsertNextPoint(input_data.GetPoint((dims[0]*dims[1]*(dims[2]-1))))
    tup_list.append(input_data.GetPoint((dims[0]*dims[1]*(dims[2]-1))))
    sampled_data_value.append(Data_Value_Array[(dims[0]*dims[1]*(dims[2]-1))])
    val_scalar.InsertNextValue(Data_Value_Array[(dims[0]*dims[1]*(dims[2]-1))])


    sampled_points.InsertNextPoint(input_data.GetPoint((dims[0]*dims[1]*(dims[2]-1))+(dims[0]-1)))
    tup_list.append(input_data.GetPoint((dims[0]*dims[1]*(dims[2]-1))+(dims[0]-1)))
    sampled_data_value.append(Data_Value_Array[(dims[0]*dims[1]*(dims[2]-1))+(dims[0]-1)])
    val_scalar.InsertNextValue(Data_Value_Array[(dims[0]*dims[1]*(dims[2]-1))+(dims[0]-1)])

    sampled_points.InsertNextPoint(input_data.GetPoint((dims[0]*dims[1]*(dims[2]))-dims[0]))
    tup_list.append(input_data.GetPoint((dims[0]*dims[1]*dims[2])-dims[0]))
    sampled_data_value.append(Data_Value_Array[(dims[0]*dims[1]*dims[2])-dims[0]])
    val_scalar.InsertNextValue(Data_Value_Array[(dims[0]*dims[1]*dims[2])-dims[0]])

    sampled_points.InsertNextPoint(input_data.GetPoint(num_points-1))
    tup_list.append(input_data.GetPoint(num_points-1))
    sampled_data_value.append(Data_Value_Array[num_points-1])
    val_scalar.InsertNextValue(Data_Value_Array[num_points-1])


# Randomly selecting rest of the sample points.
    loop = num_samples-8
    while(loop >= 1):
        x = random.randint(0, 249)
        y = random.randint(0, 249)
        z = random.randint(0, 49)
        tup = (x,y,z)
        # print(tup)
        if(tup not in tup_list):
            sampled_points.InsertNextPoint(tup)
            tup_list.append(tup)
            sampled_data_value.append(Data_Value_Array[(tup[2]*250*250) + (tup[1]*250) + tup[0]])
            val_scalar.InsertNextValue(Data_Value_Array[(tup[2]*250*250) + (tup[1]*250) + tup[0]])

            loop -= 1
        else:
            continue
    return (tup_list,sampled_points,sampled_data_value,val_scalar)


# Function to compute signal-to-noise ratio

def compute_SNR(arrgt, arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt) - np.min(arrgt))**2
    snr = 10*np.log10(sqd_max_diff/np.mean(diff**2))
    return snr



# Function to reconstruct dataset from sampled points 
 
def reconstruction(method, tup_list, Data_Value_Array):

    start = time.time()
    x = []
    y = []
    z = []
    for i in tup_list:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    z_a = np.array(x)
    y_a = np.array(y)
    x_a = np.array(z)

    xi, yi, zi = np.mgrid[min(x_a):max(x_a):50j, min(y_a):max(y_a):250j, min(z_a):max(z_a):250j] # creating grid for reconstruction

    # Nearest reconstruction method
    if (method.lower() == 'nearest'):
        nearest = griddata((x_a, y_a, z_a), np.array(sampled_data_value), (xi, yi, zi), method='nearest')

        # Create a vtkImageData object for nearest neighbor interpolation
        nearest_image = vtk.vtkImageData()
        nearest_image.SetDimensions((250,250,50))
        nearest_image.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(nearest.flatten(), deep=True))

        # Write the reconstructed volume data as a VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName('output_nearest.vti')
        writer.SetInputData(nearest_image)
        writer.Write()
        end = time.time()
        print("The reconstructed volume data from the sampled points using nearest method is written in the file named : \"output_nearest.vti\". ")
        print("Time taken for reconstruction = ", end-start)

        reader1 = vtk.vtkXMLImageDataReader()
        reader1.SetFileName("output_nearest.vti")
        reader1.Update()
        nearest_reconstruct_data = reader1.GetOutput()
        scalar_data1 = reader1.GetOutput().GetPointData().GetScalars()
        Data_Value_Array_nearest = ns.vtk_to_numpy(scalar_data1)

        signal_to_noise_nearest  = compute_SNR(Data_Value_Array,Data_Value_Array_nearest)

        print("The Signal-to-Noise ratio of this reconstruction as compared to original raw volume data = ", signal_to_noise_nearest)

    # Linear reconstruction method
    elif (method.lower() == 'linear'):
        linear = griddata((x_a, y_a, z_a), np.array(sampled_data_value), (xi, yi, zi), method='linear')
        #handling isnan values in linear method
        nan = np.isnan(linear)
        nearest = griddata((x_a, y_a, z_a), np.array(sampled_data_value), np.transpose(np.nonzero(nan)), method='nearest')
        linear[nan] = nearest


        # Create a vtkImageData object for linear interpolation
        linear_image = vtk.vtkImageData()
        linear_image.SetDimensions((250,250,50))
        linear_image.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(linear.flatten(), deep=True))


        # Write the reconstructed volume data as a VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName('output_linear.vti')
        writer.SetInputData(linear_image)
        writer.Write()
        end = time.time()
        print("The reconstructed volume data from the sampled points using linear method is written in the file named : \"output_linear.vti\". ")
        print("Time taken for reconstruction = ", end-start)

        reader2 = vtk.vtkXMLImageDataReader()
        reader2.SetFileName("output_linear.vti")
        reader2.Update()
        linear_reconstruct_data = reader2.GetOutput()
        scalar_data2 = reader2.GetOutput().GetPointData().GetScalars()
        Data_Value_Array_linear = ns.vtk_to_numpy(scalar_data2)

        signal_to_noise_linear  = compute_SNR(Data_Value_Array,Data_Value_Array_linear) # calling snr function

        print("The Signal-to-Noise ratio of this reconstruction as compared to original raw volume data = ", signal_to_noise_linear)

    else:
        print("Wrong Input: Choose one of the above mentioned reconstruction method i.e. (nearest/linear)")


# Load the input 3D dataset in .vti format
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Isabel_3D.vti")
reader.Update()
input_data = reader.GetOutput()
Data_Value_Array = ns.vtk_to_numpy(input_data.GetPointData().GetArray('Pressure'))

sampling_per = int(input("ENTER SAMPLING PERCENTAGE (percentage of data you want to be sampled) : \n"))
reconstruction_method = input("ENTER RECONSTRUCTION METHOD (nearest/linear) : \n")
# print(type(reconstruction_method))
tup_list,sampled_points,sampled_data_value,val_scalar = SRS_function(input_data,sampling_per)


# Create a polydata object to store the sampled points and data values
polydata = vtk.vtkPolyData()
polydata.SetPoints(sampled_points)
polydata.GetPointData().SetScalars(val_scalar)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("sampled_points.vtp")
writer.SetInputData(polydata)
writer.Write()

print("The output sampled points are stored in \"sampled_points.vtp\" file. It can be loaded in paraview and visualized using Point Gaussian representation.")


reconstruction(reconstruction_method,tup_list,Data_Value_Array)