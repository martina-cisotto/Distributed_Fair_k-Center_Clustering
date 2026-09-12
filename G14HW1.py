from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import time
import math

def FairFFT(U, kA, kB):
  distance = math.dist
  n = len(U)
  tot_k = kA + kB
  S = []              #list of centers

  ### INITIAL CHECKS
  #if U is empty or sum of kA kB is equal or less than zero return empty list
  if n == 0 or tot_k <= 0:
    return []

  #check if there are enough points with label A or B
  dispA = 0
  for u in U:
    if u[1] == 'A':
        dispA += 1
  dispB = n - dispA

  #if partition has less points than kA or kB, takes what is disponible
  kA_safe = min(kA, dispA)
  kB_safe = min(kB, dispB)
  tot_k_safe = kA_safe + kB_safe


  ### SELECT FIRST CENTER
  #find the first casual point which respect the conditions
  valid_points_idx = []
  for i in range(n):
    #point valid if it has label A and we need at least a center A (kA>0), same for B
    if (U[i][1] == 'A' and kA_safe > 0) or (U[i][1] == 'B' and kB_safe > 0):
      valid_points_idx.append(i)

  valid_n = len(valid_points_idx)
  #if no point respect conditions
  if valid_n == 0:
    return []

  rand.seed(42)
  idx = rand.randint(0, valid_n - 1)
  first_idx = valid_points_idx[idx]

  #append the first center
  S.append(U[first_idx])

  #initialize counters
  countA = 0
  countB = 0
  #count if the chosen point belongs to A or B
  if U[first_idx][1] == 'A':
    countA += 1
  else:
    countB += 1


  ### INITIALIZE DISTANCES
  #distances between the first center and every point in U
  dist_S = []
  for i in range(n):
    dist = distance(U[i][0], U[first_idx][0])
    dist_S.append(dist)

  #don't consider the first point (already selected)
  dist_S[first_idx] = -float('inf')


  ### MAIN LOOP to find other centers
  #continue the loop until the number of centers is less than the sum of kA and kB
  while len(S) < tot_k_safe:
    best_dist = -1
    best_idx = -1

    #find the farthest point that respect group conditions
    for i in range(n):
      d = dist_S[i]
      if d > best_dist:     #if distance between the center and the point is bigger than the initialized one
        #check the label A or B, and the amount of countA or countB respectively
        if (U[i][1] == 'A' and countA < kA_safe) or (U[i][1] == 'B' and countB < kB_safe):
          #update
          best_dist = d
          best_idx = i

    #stop if no more valid points are found
    if best_idx == -1:
      break

    #add new center and update group counters
    S.append(U[best_idx])
    dist_S[best_idx] = -float('inf')   #don't consider the point already selected

    if U[best_idx][1] == 'A':
      countA += 1
    else:
      countB += 1

    #update the min distance of each point to the centers
    for i in range(n):
      if dist_S[i] != -float('inf'):
        new_dist = distance(U[i][0], U[best_idx][0])  #compute distance between new center and points in U
        if new_dist < dist_S[i]:
          dist_S[i] = new_dist

  return S


def MRFairFFT(U_rdd, kA, kB, L):

  #ROUND 1
  def local_fairFFT(partition):

    points = list(partition)
    if len(points) == 0:
      return iter([])

    #count groups to avoid problems with kA_loc kB_loc
    countA = 0
    for p in points:
      if p[1] == 'A':
        countA += 1
    countB = len(points) - countA

    #set larger k for local partitions
    kA_loc = min(2 * kA, countA)
    kB_loc = min(2 * kB, countB)

    #if partition too small return all
    if len(points) <= kA_loc + kB_loc:
      return iter(points)

    #local coreset
    S_local = FairFFT(points, kA_loc, kB_loc)
    return iter(S_local)

  #FairFFT on each partition
  coreset_RDD = U_rdd.mapPartitions(local_fairFFT)

  #ROUND 2
  #collect all local coresets
  coreset = coreset_RDD.collect()

  #final FairFFT with respect to kA kB
  S_final = FairFFT(coreset, kA, kB)

  return S_final


def prepare_data(line):
  #cut string
  parts = line.split(',')
  #coords into numbers
  coords = tuple(float(x) for x in parts[:-1])
  group = parts[-1].strip()
  return (coords, group)

def objective_funct(inputPoints, S):
  #list of center's coords (without A or B)
  centers = []
  for c in S:
    centers.append(c[0])

  #compute dist from point to every center to find min (find distance from a point to its nearest center)
  def get_min_dist(point):
    p = point[0]
    dist = [math.dist(p, c) for c in centers]
    return min(dist)

  #apply get_min_dist to every point in U (using spark) and extract max (find max of the min distances across all points)
  return inputPoints.map(get_min_dist).max()

def main():
  if len(sys.argv) != 5:
    print('error')
    sys.exit(1)

  #input reading
  file_path = sys.argv[1]
  kA = int(sys.argv[2])
  kB = int(sys.argv[3])
  L = int(sys.argv[4])

  #control for kA kB
  if (kA+kB) <= 0:
    print(f'The sum of kA and kB must be greater than zero!')
    return   #exit without strting spark

  print(f'File path = {os.path.basename(file_path)}, KA = {kA}, KB = {kB}, L = {L}')
  #spark setup
  conf = SparkConf().setAppName('G14HW1')
  sc = SparkContext(conf=conf)

  #read file and partitioning
  inputPoints = sc.textFile(file_path).repartition(L).map(prepare_data).cache()

  #count N, NA, NB
  N = inputPoints.count()
  NA = inputPoints.filter(lambda x: x[1] == "A").count()
  NB = N - NA
  print(f'N = {N}, NA = {NA}, NB = {NB}')

  #run MRFairFFT and compute time
  start_time = time.time()
  S = MRFairFFT(inputPoints, kA, kB, L)
  end_time = time.time()
  execution_time = int((end_time - start_time) * 1000)

  for c in S:
    #remove the space in lists between the centers values
    cen_coords = "[" + ",".join(map(str, c[0])) + "]"
    print(f'Center = {cen_coords} Label = {c[1]}')

  #compute the objective_funct
  print(f'Objective function = {objective_funct(inputPoints, S)}')

  print(f'Running time of MRFairFFT = {execution_time} ms')


if __name__ == "__main__":
    main()