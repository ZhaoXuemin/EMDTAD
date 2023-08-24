#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Class   :
# @description:
# @Time    : 2021/11/19 下午5:05
# @Author  : xuemin Zhao

import numpy as np
import sys
from PyEMD import EMD
from skimage.filters.rank import entropy
from skimage.morphology import disk
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Get EMD_TAD arguments")
    parser.add_argument("Input_path", type=str, help="Input matrix path")
    parser.add_argument("Result_path", type=str, help="Result save path")
    parser.add_argument("Imf_deep", type=int, help="The deep of imf")
    parser.add_argument("Diagonal_deep", type=int, help="Diagonal depth of reconstruction")

    args = parser.parse_args()
    return args.Input_path, args.Result_path, args.Imf_deep, args.Diagonal_deep

def normalization(data):
    _range = np.max(data)-np.min(data)
    return (data - np.min(data))/_range

def zuHeIndex(li,imf_deep):
    reli = []
    for i in range(0, len(li)):
        if i == 0:
            reli.append([li[i]])
        else:
            addli = []
            addli.append([li[i]])
            for ii in reli:
                addli.append(ii+[li[i]])
                addli.append( [li[i]] + ii)
            reli += addli
    fin = []
    for vv in reli:
        if len(vv) == imf_deep:
            fin.append(vv)

    return fin


def change_diag_vale(ice_data, new_value, index):
    x_index = index
    y_index = 0
    for value in new_value:
        ice_data[x_index, y_index] = value
        x_index += 1
        y_index += 1
    return ice_data


def get_new_value(sp_value, imf_deep,imf_value):
    emd = EMD()
    emd.emd(sp_value, max_imf=imf_deep)
    imfs, res = emd.get_imfs_and_residue()
    finale = sp_value
    imf_index = 0
    for imf in imfs:
        finale = finale + (imf*imf_value[imf_index])
        imf_index += 1
    return finale


def get_mat_Entropy(input_mat):
    greyIm = input_mat
    N = 20
    S = greyIm.shape
    E = np.array(greyIm)
    for row in range(S[0]):
        for col in range(S[1]):
            Lx = np.max([0, col - N])
            Ux = np.min([S[1], col + N])
            Ly = np.max([0, row - N])
            Uy = np.min([S[0], row + N])
            region = greyIm[Ly:Uy, Lx:Ux].flatten()
            E[row, col] = entropy(region)
    return np.sum(E)


def get_imfmatrix_from_oringlmatrix(matrix_data, imf_deep =3, diagonal_deep= 30):
    # get imf matrix
    print('Find best coefficient')
    zuhe = zuHeIndex(np.arange(0.5, 1.1, 0.1), imf_deep)
    entro_list = []
    bili_list = []
    zuhe_index = 0
    for value_list in zuhe:
        demo_mat = matrix_data[100:200, 100:200]
        zores_mat = np.zeros(np.shape(demo_mat))
        for ofindex in range(diagonal_deep):
            diag_value = np.diagonal(demo_mat, offset=ofindex)
            imf_value = get_new_value(diag_value,imf_deep,value_list)
            zores_mat = change_diag_vale(zores_mat, imf_value, ofindex)
        re = np.tril(zores_mat)
        re = re + re.T - np.diag(np.diag(re))
        entropy_value = np.sum(entropy(normalization(re), disk(20)))
        entro_list.append(entropy_value)
        bili_list.append(value_list)
        zuhe_index +=1
    index_flag = np.max(np.array(entro_list))
    index = 0
    final_list = []
    for env in entro_list:
        if env == index_flag:
            final_list.append(bili_list[index])
        index += 1


    for ofindex in range(diagonal_deep):
        diag_value = np.diagonal(matrix_data, offset=ofindex)
        imf_value = get_new_value(diag_value, imf_deep, final_list[-1])
        matrix_data = change_diag_vale(matrix_data, imf_value, ofindex)
    re = np.tril(matrix_data)
    re = re + re.T - np.diag(np.diag(re))

    return re


def recover_zore_colum(matrix):
    count = 0
    countlist = []
    for lin in matrix:
        if np.sum(lin) == 0:
            countlist.append(count)
        count += 1
    return countlist


def matrix_tansform(matrix_path,imf_deep):
    ice_data = np.loadtxt(matrix_path)
    countlist = recover_zore_colum(ice_data)
    result = get_imfmatrix_from_oringlmatrix(ice_data, imf_deep, 100)
    print(countlist)
    for lin in countlist:
        result[lin:lin + 1, :] = 0
        result[:, lin:lin + 1] = 0

    result[result < 0] = 0
    return result


def get_dig_imfs(sp_value, imf_deep):
    emd = EMD()
    emd.emd(sp_value, max_imf=imf_deep)
    imfs, res = emd.get_imfs_and_residue()
    flag = 0
    for imf in imfs:
        print(type(imf))
        imf[imf > 200] = 0
        imf[imf < -200] = 0
        np.savetxt('./dia/imf{0}.txt'.format(flag), imf, fmt='%1.3f')
        flag += 1


def unaddimfs(matrix_path, imf_deep):
    ice_data = np.loadtxt(matrix_path)
    countlist = recover_zore_colum(ice_data)


def get_unaddimfs_from_oringlmatrix(matrix_path, imf_deep, diagonal_deep):
    matrix_data = np.loadtxt(matrix_path)
    emd = EMD()
    allre = np.full((imf_deep, 4864, 4864), 0)
    for ofindex in range(diagonal_deep):
        diag_value = np.diagonal(matrix_data, offset=ofindex)
        emd.emd(diag_value)
        imfs, res = emd.get_imfs_and_residue()
        for ma_index in range(imf_deep):
            allre[ma_index] = change_diag_vale(allre[ma_index],imfs[ma_index],ofindex)
        print(ofindex)

    for ma_index in range(imf_deep):
        rr = allre[ma_index]
        re = np.tril(rr)
        re = re + re.T - np.diag(np.diag(re))
        np.savetxt('/Volumes/Samsung_X5/EMDHIC/dpn/unaddimf/emdimf{0}'.format(ma_index),re)


def emd_Cluster(imf_matrix,proc=1.15):
    binrange = 5
    bias = 0.01
    emdshape = np.shape(imf_matrix)
    emdCount = np.zeros(emdshape[0])
    matrix = imf_matrix
    for index in range(binrange,emdshape[0]-binrange):
        point = index
        uavalue = 0
        davalue = 0
        for i in range(1, 3):
            ua = point - i
            uavalue = uavalue + np.sum(matrix[ua - 4:ua, ua + 1:ua + 5]) / 16
        cvalue = np.sum(matrix[point - 4:point, point + 1:point + 5]) / 16
        for i in range(1, 3):
            na = point + i
            davalue = davalue + np.sum(matrix[na - 4:na, na + 1:na + 5]) / 16
        sum_value = davalue + uavalue + bias
        sum_value = cvalue / sum_value
        if sum_value < 0:
            sum_value = 0
        emdCount[index] = sum_value

    relist = []
    valuelist = []
    for index in range(1,len(emdCount)-1):
        if emdCount[index] < emdCount[index-1]:
            if emdCount[index] < emdCount[index +1]:
                if emdCount[index] < np.mean(emdCount):
                    relist.append(index)
                    valuelist.append(emdCount[index])

    en = np.mean(np.array(valuelist))*proc
    fin =[1]
    bin_index = 0
    for vv in valuelist:
        if vv < en:
            fin.append(relist[bin_index])
        bin_index +=1
    fin.append(emdshape[0])

    return fin


def count_CI_value(matrix_path,celline):
    print(matrix_path)
    np.loadtxt()



if __name__ == "__main__":

    input_path, output_path, imf_deep, diagonal_deep = get_args()
    print('1:Load input File')
    input_matrix = np.loadtxt(input_path)
    print('Load Success')

    print('2: Reconstruct the matrix by EMD')
    re = get_imfmatrix_from_oringlmatrix(input_matrix, imf_deep, diagonal_deep)
    print('Reconstruct Success')

    print('3: Detection of TAD')
    relist = emd_Cluster(re)
    print('4: Save Result')
    np.savetxt(output_path, relist, fmt='%i')















