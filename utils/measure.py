import math

from itertools import combinations, permutations;

def calculate_distance(point1, point2):
    # 두 점 사이의 유클리드 거리 계산
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def rectangle_line_segments(width, height):
    # 사각형의 꼭짓점 정의 (원점을 기준으로 한 사각형)
    vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    verticesLabel = [0,1,2,3]
    
    # 모든 꼭짓점 쌍의 조합을 계산 (2개씩)
    line_combinations = list(permutations(verticesLabel, 2))
    
    # 각 선분의 길이를 계산하여 저장
    line_lengths = [(combo[0], combo[1], calculate_distance(vertices[combo[0]], vertices[combo[1]])) for combo in line_combinations]
    
    return line_lengths

def get_square_distance_map(width,height):
  lines = rectangle_line_segments(width,height)
  print(lines)
  distanceMap = dict()
  for line in lines:
    if(not distanceMap.get(line[0])):
       distanceMap[line[0]] = {}
    distanceMap[line[0]][line[1]] = line[2]
    # pass

  print(distanceMap)
  return distanceMap



if __name__ == "__main__":
   get_square_distance_map()
   