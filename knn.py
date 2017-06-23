# coding=utf-8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# KNN 거리 계산 ------------------------------------------------
def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - data_set  # Input_vct는 data_set 아이소타입 및 빼기로 확대
    sq_diff_mat = diff_mat**2  # 매트릭스 사각형의 각 요소
    distance = sq_diff_mat.sum(axis=1)**0.5  # 각 행과 제곱근을 합산
    return distance.min(axis=0)  # 가장 작은 거리를 돌려줍니다

# csv 2 matrix --------------------------------------------------
def file2mat(test_filename, para_num):
    """
    매트릭스 테이블로의 열 수, test_filename 경로를 형성하기 위해 증착된다 para_num 행렬
     반환 대상 행렬, 행렬 및 데이터 카테고리의 각 행 
    """
    fr = open(test_filename)        # csv 파일 열고
    lines = fr.readlines()          # 라인별로 읽어서
    line_nums = len(lines)          # 총 라인수
    # para_num 은 원하는 필드, line_num은 레코드 번호
    result_mat = np.zeros((line_nums, para_num))  # 매트릭스 line_nums 행, 열 para_num 만들기

    class_label = []
    # 레코드별로 
    for i in range(line_nums):
        line = lines[i].strip()                     # 숫자 제거
        item_mat = line.split(',')                  # 컴마 로 분리
        result_mat[i, :] = item_mat[0: para_num]    # 필드별로 분리
        class_label.append(item_mat[-1])  # class_label 추출
    fr.close()                      # csv 파일 닫고
    return result_mat, class_label  # 매트릭스/레이블 리턴

# calc ROC curve -------------------------------------------------- 
def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    roc_rate = np.zeros((2, data_set_size))
    for i in range(data_set_size):
        if data_set[2][i] == 1:
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(1000):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 2:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal  # 일반 점 / 점 모두 정상에게 위의 임계 값
        roc_rate[1][j] = abnormal1 / abnormal  # 임계 값 위의 아웃 라이어 / 모든 이상치
    return roc_rate

# classification test & Scatter + ROC Plot --------------------------------------
def test(training_filename, test_filename):
    # 학습쌍, 32개의 필드
    training_mat, training_label = file2mat(training_filename, 32)
    # 검증쌍
    test_mat, test_label = file2mat(test_filename, 32)
    # 검증쌍 갯수
    test_size = test_mat.shape[0]
    result = np.zeros((test_size, 3))
    # 수, 최소 유클리드 거리, 상기 테스트 데이터는 카테고리 설정
    for i in range(test_size):
        result[i] = i + 1, classify(test_mat[i], training_mat), test_label[i]  
    result = np.transpose(result)  # 행렬 전치
    plt.figure(1)
    plt.scatter(result[0], result[1], c=result[2], edgecolors='None', s=2, alpha=1)
    # 검증쌍 순서에 따라, 가로축은 데이타순서, 세로축은 최소 유클리드 거리, 레이블 색상
    # 도 산포도 1. 횡축 수를 나타낸다는, 세로축이 카테고리에서 테스트 세트의 데이터에 따라 상기 최소 유클리드 거리, 
    # 색의 중심점을 나타내고, 상기 주변 노 컬러 도트 (1)의 최소 스폿 사이즈, 최대 계조
    roc_rate = roc(result)
    plt.figure(2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=2, alpha=1)
    # 도 임계 지점 정상 / 보통 요점 즉 상기 2 ROC 곡선 횡축 오경보 율;. 길이 감지 율, 즉 임계 특이 위 / 모든 특이
    plt.show()

# main code ----------------------------------------
if __name__ == "__main__":
    test('training.csv', 'test.csv')
