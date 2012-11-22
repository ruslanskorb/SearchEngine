//
//  main.cpp
//  SearchEngine
//
//  Created by Ruslan Skorb on 11/4/12.
//  Copyright (c) 2012 Ruslan Skorb. All rights reserved.
//

#include <cstdio>
#include <vector>
#include <cmath>

using namespace std;

void outputTripleVectorTo(FILE* out, vector< vector< vector<int> > > v)
{
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            for (int l = 0; l < v[i][j].size(); l++) {
                fprintf(out, "%d ", v[i][j][l]);
            }
            fprintf(out, "\n");
        }
        fprintf(out, "\n\n");
    }
}

void outputDualVectorTo(FILE* out, vector< vector<float> > v)
{
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            fprintf(out, "%f ", v[i][j]);
        }
        fprintf(out, "\n");
    }
}

void outputDualVectorTo(FILE* out, vector< vector<int> > v)
{
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            fprintf(out, "%d ", v[i][j]);
        }
        fprintf(out, "\n");
    }
}

void outputVectorTo(FILE *out, vector<float> v)
{
    for (int i = 0; i < v.size(); i++) {
        fprintf(out, "%f ", v[i]);
    }
}

void outputVectorTo(FILE *out, vector<int> v)
{
    for (int i = 0; i < v.size(); i++) {
        fprintf(out, "%d ", v[i]);
    }
}

vector < vector<float> > sortedVectorByColumn(vector< vector<float > > v, int column)
{
    bool is;
    do {
        is = false;
        for(int r = 0; r < v.size() - 1; r++) {
            if(v[r][column] > v[r + 1][column]) {
                for(int c = 0; c < v[r].size(); c++)
                    swap(v[r][c], v[r + 1][c]);
                is = true;
            }
        }
    } while(is);
    
    return v;
}

vector< vector<int> > vectorDFrom(FILE *in)
{
    int m, n;
    
    fscanf(in, "%d", &m);
	fscanf(in, "%d", &n);
    
    vector< vector<int> > D;
    
    for(int i = 0; i < m; i++)
    {
        vector<int> columns;
        int columnValue;
		for(int j = 0; j < n; j++)
        {
			fscanf(in, "%d", &columnValue);
            columns.push_back(columnValue);
        }
        D.push_back(columns);
    }
    
    return D;
}

vector< vector<float> > vectorS(vector< vector<int> > D)
{
    float *sumRows = new float[D.size()] {0};
    
	for(int i = 0; i < D.size(); i++)
	{
		for(int j = 0; j < D[i].size(); j++)
			sumRows[i] += D[i][j];
	}
    
    vector< vector<float> > S;
    
	for(int i = 0; i < D.size(); i++)
	{
        vector<float> columns;
		for(int j = 0; j < D[i].size(); j++)
		{
			columns.push_back(D[i][j] / ((float)(sumRows[i])) );
		}
        S.push_back(columns);
	}
    
    return S;
}

vector< vector<float> > vectorS2(vector< vector<int> > D)
{
    float *sumColumns = new float[D[0].size()] {0};
    
	for(int i = 0; i < D.size(); i++)
	{
		for(int j = 0; j < D[i].size(); j++)
			sumColumns[j] += D[i][j];
	}
    
    vector< vector<float> > S2;
    
	for(int i = 0; i < D.size(); i++)
	{
        vector<float> columns;
		for(int j = 0; j < D[i].size(); j++)
		{
			columns.push_back(D[i][j] / ((float)(sumColumns[j])) );
		}
        S2.push_back(columns);
	}
    
    return S2;
}

vector< vector<float> > transponseVector(vector< vector<float> > v)
{
    vector< vector<float> > vTransponse(v[0].size(), vector<float>(v.size()));
    
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v[0].size(); ++j)
            vTransponse[j][i] = v[i][j];
    
    return vTransponse;
}

vector< vector<int> > transponseVector(vector< vector<int> > v)
{
    vector< vector<int> > vTransponse(v[0].size(), vector<int>(v.size()));
    
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v[0].size(); ++j)
            vTransponse[j][i] = v[i][j];
    
    return vTransponse;
}

vector< vector<float> > vectorC(vector< vector<float> > S, vector< vector<float> > S2)
{
    vector< vector<float> > C;
    
	for(int i = 0; i < S.size(); i++)
	{
        vector<float> columns;
		for(int j = 0; j < S.size(); j++)
		{
            float columnValue = 0;
			for(int k = 0; k < S2.size(); k++)
			{
				columnValue += S[i][k] * S2[k][j];
			}
            columns.push_back(columnValue);
		}
        C.push_back(columns);
	}
    
	return C;
}

vector< vector<float> > vectorC2(vector< vector<float> > S, vector< vector<float> > S2)
{
    vector< vector<float> > C2;
    
	for(int i = 0; i < S2.size(); i++)
	{
        vector<float> columns;
		for(int j = 0; j < S2.size(); j++)
		{
            float columnValue = 0;
			for(int k = 0; k < S.size(); k++)
			{
				columnValue += S2[i][k] * S[k][j];
			}
            columns.push_back(columnValue);
		}
        C2.push_back(columns);
	}
    
	return C2;
}

vector<float> vectorPhi(vector< vector<float> > C)
{
    vector<float> phi;
    
	for(int i = 0; i < C.size(); i++)
	{
		phi.push_back(1 - C[i][i]);
	}
    
    return phi;
}

float deltaValue(vector< vector<float> > C)
{
    float delta = 0.0;

    for(int i = 0; i < C.size(); i++)
	{
		delta += C[i][i] / C.size();
	}
    
    return delta;
}

float phiValue(float deltaValue)
{
    float phiValue = 1 - deltaValue;
    return phiValue;
}

float numberOfClustersValue(float deltaValue, int numberOfDocumentsValue)
{
    float numberOfClustersValue = deltaValue * numberOfDocumentsValue;
    return numberOfClustersValue;
}

float averageNumberOfDocumentsInClusterValue(float deltaValue)
{
    float averageNumberOfDocumentsInClusterValue = 1 / deltaValue;
    return averageNumberOfDocumentsInClusterValue;
}

vector<float> vectorP(vector< vector<float> > C, vector<float> Phi, vector< vector<int> > D)
{
    float *sumRows = new float[D.size()] {0};
    
	for(int i = 0; i < D.size(); i++)
	{
		for(int j = 0; j < D[i].size(); j++)
			sumRows[i] += D[i][j];
	}
    
    vector<float> p;
    
    for(int i = 0; i < D.size(); i++)
	{
		p.push_back(C[i][i] * Phi[i] * sumRows[i]);
	}
    
    return p;
}

vector<int> vectorKernels(vector<float> P, int numberOfClustersV)
{
    vector< vector<float> > sortedP;
    
	for(int i = 0; i < P.size(); i++)
	{
        vector<float> pElements;
        pElements.push_back((float)i);
        pElements.push_back(P[i]);
        
        sortedP.push_back(pElements);
	}
    
    sortedP = sortedVectorByColumn(sortedP, 1);
    
    vector<int> vectorKernels;
    
	int indexOfKernel = 0;
	for(int i = ((int)(P.size())-1); i > ((int)(P.size())-1) - numberOfClustersV; i--)
	{
		vectorKernels.push_back(sortedP[i][0]);
		indexOfKernel++;
	}
    
    return vectorKernels;
}

bool isKernel(int indexOfDocument, vector<int> kernels)
{
	bool isKernel = false;
    
	for(int i = 0; i < kernels.size(); i++)
	{
		if(kernels[i] == indexOfDocument)
		{
			isKernel = true;
			break;
		}
	}
    
	return isKernel;
}

vector< vector<int> > vectorClusters(vector< vector<float> > C, vector<int> kernels, vector<float> P)
{
    vector< vector<int> > unsortedElementsOfClusters;
    
	int indexOfDocument = 0;
	do
	{
		if(isKernel(indexOfDocument, kernels) == false)
		{
            vector<int> elementOfCluster;
			float maxCoverageRatio = -1;
			int kernel = -1;
			for(int i = 0; i < kernels.size(); i++)
			{
				if(C[indexOfDocument][kernels[i]] > maxCoverageRatio)
				{
					maxCoverageRatio = C[indexOfDocument][kernels[i]];
					kernel = kernels[i];
				}
				else if (C[indexOfDocument][kernels[i]] == maxCoverageRatio)
				{
					if(P[kernels[i]] > P[kernel])
					{
						maxCoverageRatio = C[indexOfDocument][kernels[i]];
						kernel = kernels[i];
					}
				}
			}
            elementOfCluster.push_back(indexOfDocument);
            elementOfCluster.push_back(kernel);
            
            unsortedElementsOfClusters.push_back(elementOfCluster);
		}
		indexOfDocument++;
	} while(indexOfDocument < C.size());
    
	vector< vector<int> > clusters;
    
	for(int i = 0; i < kernels.size(); i++)
	{
		vector <int> elementsOfCluster;
		elementsOfCluster.push_back(kernels[i]);
		for(int j = 0; j < unsortedElementsOfClusters.size(); j++)
		{
			if(unsortedElementsOfClusters[j][1] == kernels[i])
			{
				elementsOfCluster.push_back(unsortedElementsOfClusters[j][0]);
			}
		}
		clusters.push_back(elementsOfCluster);
	}
    
    return clusters;
}

vector<float> vectorDelta(vector< vector<float> > C2)
{
    vector<float> vDelta;
    
	for(int i = 0; i < C2.size(); i++)
	{
        vDelta.push_back(C2[i][i]);
	}
    
    return vDelta;
}

vector< vector<int> > vectorFrequencyOfTermsInClusters(vector< vector<int> > clusters, vector< vector<int> > D)
{
    vector< vector<int> > frequencyOfTermsInClusters;
    
	for(int i = 0; i < clusters.size(); i++)
	{
        vector<int> frequencyOfTermsInCluster;
		int fjk = 0;
		for(int k = 0; k < D[i].size(); k++)
		{
			for(int j = 0; j < clusters[i].size(); j++)
			{
				fjk += D[clusters[i][j]][k];
			}
			frequencyOfTermsInCluster.push_back(fjk);
			fjk = 0;
		}
        frequencyOfTermsInClusters.push_back(frequencyOfTermsInCluster);
	}
    
    return frequencyOfTermsInClusters;
}

vector<int> vectorTotalNumberOfTerms(vector< vector<int> > D)
{
    vector<int> vectorTotalNumberOfTerms;
    
	for(int i = 0; i < D[0].size(); i++)
	{
        vectorTotalNumberOfTerms.push_back(0);
		for(int j = 0; j < D.size(); j++)
		{
			if(D[j][i] == 1)
				vectorTotalNumberOfTerms[i] += 1;
		}
	}
    
    return vectorTotalNumberOfTerms;
}

vector<int> vectorNumberOfClustersContainTerm(vector< vector<int> > D, vector< vector<int> > clusters)
{
    vector<int> numberOfClustersContainTerm;
    
	for(int i = 0; i < D[0].size(); i++)
	{
        numberOfClustersContainTerm.push_back(0);
		for(int l = 0; l < clusters.size(); l++)
		{
			for(int t = 0; t < clusters[l].size(); t++)
			{
				if(D[clusters[l][t]][i] == 1)
				{
					numberOfClustersContainTerm[i] += 1;
					break;
				}
			}
		}
	}
    
    return numberOfClustersContainTerm;
}

vector<float> vectorAverageFrequencyOfTermsInClusters(vector<int> totalNumberOfTerms, vector<int> numberOfClustersContainTerm)
{
    vector<float> vectorAverageFrequencyOfTermsInClusters;
    
	for(int i = 0; i < totalNumberOfTerms.size(); i++)
	{
        vectorAverageFrequencyOfTermsInClusters.push_back(totalNumberOfTerms[i] / ((float)(numberOfClustersContainTerm[i])));
	}
    
    return vectorAverageFrequencyOfTermsInClusters;
}

vector< vector<int> > vectorCentroids(vector< vector<int> > clusters, vector< vector<int> > frequencyOfTermsInClusters, vector<float> vDelta, vector<float>averageFrequencyOfTermsInClusters, float delta2V)
{
    vector< vector<int> > vectorCentroids;
    
	for(int i = 0; i < clusters.size(); i++)
	{
		vector<int> vectorCentroid;
		for(int j = 0; j < averageFrequencyOfTermsInClusters.size(); j++)
		{
			if(frequencyOfTermsInClusters[i][j] * vDelta[j] >= averageFrequencyOfTermsInClusters[j] * delta2V * 0.5)
			{
				vectorCentroid.push_back(1);
			}
			else
			{
				vectorCentroid.push_back(0);
			}
		}
        vectorCentroids.push_back(vectorCentroid);
	}
    
    return vectorCentroids;
}

vector< vector<int> > vectorClustersFromVectorD(FILE *out, vector< vector<int> > D)
{
    fprintf(out, "\n\nVector D:\n\n");
    outputDualVectorTo(out, D);
    
    fprintf(out, "\nVector S:\n\n");
    vector< vector<float> > S = vectorS(D);
    outputDualVectorTo(out, S);
    
    fprintf(out, "\nVector S2:\n\n");
    vector< vector<float> > S2 = vectorS2(D);
    outputDualVectorTo(out, S2);
    
	fprintf(out, "\nTransponsed vector S2:\n\n");
    vector< vector<float> > transponsedS2 = transponseVector(S2);
    outputDualVectorTo(out, transponsedS2);
    
    fprintf(out, "\nVector C:\n\n");
    vector< vector<float> > C = vectorC(S, transponsedS2);
    outputDualVectorTo(out, C);
    
    fprintf(out, "\nVector C2:\n\n");
    vector< vector<float> > C2 = vectorC2(S, transponsedS2);
    outputDualVectorTo(out, C2);
    
    fprintf(out, "\nVector Phi:\n\n");
    vector<float> Phi = vectorPhi(C);
    outputVectorTo(out, Phi);
    
    fprintf(out, "\n\nDelta value:\n\n");
    float deltaV = deltaValue(C);
    fprintf(out, "%f", deltaV);
    
    fprintf(out, "\n\nPhi value:\n\n");
    float phiV = phiValue(deltaV);
    fprintf(out, "%f", phiV);
    
    fprintf(out, "\n\nNumber of clusters value:\n\n");
    float numberOfClustersV = numberOfClustersValue(deltaV, (int)(D.size()));
    fprintf(out, "%f", numberOfClustersV);
    
    fprintf(out, "\n\nAverage number of documents in cluster value:\n\n");
    float averageNumberOfDocumentsInClusterV = averageNumberOfDocumentsInClusterValue(deltaV);
    fprintf(out, "%f", averageNumberOfDocumentsInClusterV);
    
    fprintf(out, "\n\nVector collecting abilities:\n\n");
    vector<float> P = vectorP(C, Phi, D);
    outputVectorTo(out, P);
    
    fprintf(out, "\n\nVector kernels:\n\n");
    vector<int> kernels = vectorKernels(P, numberOfClustersV);
    outputVectorTo(out, kernels);
    
    fprintf(out, "\n\nClusters:\n\n");
    vector< vector<int> > clusters = vectorClusters(C, kernels, P);
    outputDualVectorTo(out, clusters);
    
    return clusters;
}

vector< vector<int> > vectorCentroids(FILE *out, vector< vector<int> > D, vector< vector<int> > clusters)
{
    vector< vector<float> > S = vectorS(D);
    vector< vector<float> > S2 = vectorS2(D);
    vector< vector<float> > transponsedS2 = transponseVector(S2);
    vector< vector<float> > C = vectorC(S, transponsedS2);
    vector< vector<float> > C2 = vectorC2(S, transponsedS2);
    
    fprintf(out, "\n\nVector delta:\n\n");
    vector<float> vDelta = vectorDelta(C2);
    outputVectorTo(out, vDelta);
    
    fprintf(out, "\n\nDelta2 value:\n\n");
    float delta2V = deltaValue(C2);
    fprintf(out, "%f", delta2V);
    
    fprintf(out, "\n\nFrequency of terms in clusters:\n\n");
    vector< vector<int> > frequencyOfTermsInClusters = vectorFrequencyOfTermsInClusters(clusters, D);
    outputDualVectorTo(out, frequencyOfTermsInClusters);
    
    fprintf(out, "\n\nTotal number of terms:\n\n");
    vector<int> totalNumberOfTerms = vectorTotalNumberOfTerms(D);
    outputVectorTo(out, totalNumberOfTerms);
    
    fprintf(out, "\n\nThe number of clusters that contain the term:\n\n");
    vector<int> numberOfClustersContainTerm = vectorNumberOfClustersContainTerm(D, clusters);
    outputVectorTo(out, numberOfClustersContainTerm);
    
    fprintf(out, "\n\nThe average frequency of terms in the clusters:\n\n");
    vector<float> averageFrequencyOfTermsInClusters = vectorAverageFrequencyOfTermsInClusters(totalNumberOfTerms, numberOfClustersContainTerm);
    outputVectorTo(out, averageFrequencyOfTermsInClusters);
    
    fprintf(out, "\n\nCentroids:\n\n");
    vector< vector<int> > vCentroids = vectorCentroids(clusters, frequencyOfTermsInClusters, vDelta, averageFrequencyOfTermsInClusters, delta2V);
    outputDualVectorTo(out, vCentroids);
    
    return vCentroids;
}

vector< vector<int> > optimizedVectorCentroid(vector< vector<int> > vCentroids)
{
    vector< vector<int> > optimizedVCentroid = vCentroids;
    vector< vector<int> > transponseOptimizedVCentroid = transponseVector(optimizedVCentroid);
    optimizedVCentroid.clear();
    
    for (int i = 0; i < transponseOptimizedVCentroid.size(); i++) {
        bool isInsignificantDocument = true;
        for (int j = 0; j < transponseOptimizedVCentroid[i].size(); j++) {
            if (transponseOptimizedVCentroid[i][j] == 1) {
                isInsignificantDocument = false;
                break;
            }
        }
        if (isInsignificantDocument == false) {
            optimizedVCentroid.push_back(transponseOptimizedVCentroid[i]);
        }
    }
    
    return transponseVector(optimizedVCentroid);
}

int summOfElementsOfVector(vector<int> v)
{
    int sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum;
}

vector<int> vectorOfSummOfTermsInVectorD(vector< vector<int> > D)
{
    vector<int> v;
    
    for (int i = 0; i < D.size(); i++) {
        v.push_back(summOfElementsOfVector(D[i]));
    }
    
    return v;
}

vector< vector<int> >vectorDG(vector< vector<int> > D,vector<int> centroid)
{
    vector< vector<int> > DG;
    
    for (int i = 0; i < D.size(); i++) {
        vector <int> v;
        for (int j = 0; j < D[i].size(); j++) {
            int l = centroid[i] == 1 ? 1 : 2;
            v.push_back(l == D[i][j] ? 1 : 0);
        }
        DG.push_back(v);
    }
    
    return DG;
}

vector<float> vectorSD(vector< vector<int> > DG, vector< vector<int> > D, vector<int> centroid)
{
    vector<float> SD;
    
    int sumCentroid = summOfElementsOfVector(centroid);
    
    for (int i = 0; i < DG.size(); i++) {
        int sumDi = summOfElementsOfVector(D[i]);
        int sumDGi = summOfElementsOfVector(DG[i]);
        
        float SDi = 2 * sumDGi / (sumDi + (float)(sumCentroid));
        
        SD.push_back(SDi);
    }
    
    return SD;
}

float maxValueSD(vector<float> SD)
{
    float maxValue = INT32_MIN;
    
    for (int i = 0; i < SD.size(); i++) {
        if(SD[i] > maxValue) {
            maxValue = SD[i];
        }
    }
    
    return maxValue;
}

int indexOfDocumentWithMaxSD(vector<float> SD, float maxSD)
{
    int indexOfDocumentWithMaxSD = 0;
    
    for (int i = 0; i < SD.size(); i++) {
        if (SD[i] == maxSD) {
            indexOfDocumentWithMaxSD = i;
            break;
        }
    }
    
    return indexOfDocumentWithMaxSD;
}

vector<int> vectorSearchTerms(int termsCount)
{
    vector<int> terms;
    
    int indexOfTerm = -2;
    do {
        printf("Enter index of term (break: '-1'): ");
        scanf("%d", &indexOfTerm);
        if (indexOfTerm > -1 && indexOfTerm < (termsCount - 1) ) {
            terms.push_back(indexOfTerm);
        } else if(indexOfTerm != -1) {
            printf("Incorrect index!\n");
        }
    } while (indexOfTerm != -1);
    
    return terms;
}

int indexOfVectorCentroidsForSearchTerms(vector<int> searchTerms, vector< vector<int> > vCentroids)
{
    vector<int> vSearchTerms;
    
    for (int i = 0; i < vCentroids[0].size(); i++) {
        vSearchTerms.push_back(0);
    }
    
    for (int i = 0; i < searchTerms.size(); i++) {
        vSearchTerms[searchTerms[i]] = 1;
    }
    
    double sumProduct = 0.0;
    double sumSQSearch = 0.0;
    double sumSQCentroid = 0.0;
    double cos;
    double maxCos = -100000.0;
    int indexOfCentroidForSearchTerms = -1;
    
    for (int i = 0; i < vCentroids.size(); i++) {
        for (int j = 0; j < vSearchTerms.size(); j++) {
            sumProduct      += vSearchTerms[j]   * vCentroids[i][j];
            sumSQSearch     += vSearchTerms[j]   * vSearchTerms[j];
            sumSQCentroid   += vCentroids[i][j]  * vCentroids[i][j];
        }
        cos = sumProduct / ( sqrt(sumSQSearch) * sqrt(sumSQCentroid) );
        printf("%f\n", cos);
        if (cos > maxCos) {
            maxCos = cos;
            indexOfCentroidForSearchTerms = i;
        }
    }
    
    return indexOfCentroidForSearchTerms;
}

vector<int> generalVectorCentroid(vector< vector<int> > D)
{
    vector<int> generalV;
    
    vector<int> summTermsInAllDocuments;
    int summ = 0;
    
    for (int i = 0; i < D[0].size(); i++) {
        for (int j = 0; j < D.size(); j++) {
            summ += D[j][i];
        }
        summTermsInAllDocuments.push_back(summ);
        summ = 0;
    }
    
    int totalSumTermsInAllDocuments = summOfElementsOfVector(summTermsInAllDocuments);
    
    float avg = totalSumTermsInAllDocuments / ( (float)(summTermsInAllDocuments.size()) );
    
    for (int i = 0; i < summTermsInAllDocuments.size(); i++) {
        if (summTermsInAllDocuments[i] > avg) {
            generalV.push_back(1);
        } else {
            generalV.push_back(0);
        }
    }
    
    return generalV;
}

vector< vector<int> > vectorCentroidsForSearchTerms(vector< vector< vector<int> > > vTreeCentroids, int indexOfVectorCentroids, vector<int> vectorClusters)
{
    vector< vector<int> > vCentroidsForSearchTerms;
    
    for (int i = 0; i < vectorClusters.size(); i++) {
        vCentroidsForSearchTerms.push_back(vTreeCentroids[indexOfVectorCentroids][vectorClusters[i]]);
    }
    
    return vCentroidsForSearchTerms;
}

vector< vector<int> > vectorClustersForSearchTerms(vector< vector< vector<int> > > vTreeClusters, int indexOfVectorCentroids, vector<int> vectorClusters)
{
    vector< vector<int> > vClustersForSearchTerms;
    
    for (int i = 0; i < vectorClusters.size(); i++) {
        vClustersForSearchTerms.push_back(vTreeClusters[indexOfVectorCentroids][vectorClusters[i]]);
    }
    
    return vClustersForSearchTerms;
}

int main()
{
	FILE *in = fopen("/Users/ruslan/Developer/SearchEngine/SearchEngine/input.txt", "r");
    vector< vector<int> > D = vectorDFrom(in);
    fclose(in);
    
	FILE *out = fopen("/Users/ruslan/Developer/SearchEngine/SearchEngine/output.txt", "w");
    
    fprintf(out, "\n\nSearch terms:\n\n");
    vector<int> searchTerms = vectorSearchTerms( (int)(D[0].size()) );
    outputVectorTo(out, searchTerms);
    
    fprintf(out, "\n\nGeneral vector centroid:\n\n");
    vector<int> generalVCentroid = generalVectorCentroid(D);
    outputVectorTo(out, generalVCentroid);
    
    vector< vector< vector<int> > > vTreeCentroids;
    vector< vector< vector<int> > > vTreeClusters;
    
    vector< vector<int> > vDocuments;
    
    for (int i = 0; i < D.size(); i++) {
        vector<int> v;
        v.push_back(i);
        vDocuments.push_back(v);
    }
    
    vTreeClusters.push_back(vDocuments);
    vTreeCentroids.push_back(D);
    
    vector< vector<int> > vClusters     = vectorClustersFromVectorD(out, optimizedVectorCentroid(D));
    vector< vector<int> > vCentroids    = vectorCentroids(out, optimizedVectorCentroid(D), vClusters);
    vTreeClusters.push_back(vClusters);
    vTreeCentroids.push_back(vCentroids);
    
    while (vCentroids.size() > 2) {
        D = vCentroids;
        vClusters = vectorClustersFromVectorD(out, optimizedVectorCentroid(D));
        vCentroids = vectorCentroids(out, optimizedVectorCentroid(D), vClusters);
        vTreeClusters.push_back(vClusters);
        vTreeCentroids.push_back(vCentroids);
    }
    
    fprintf(out, "\n\nTree centroids:\n\n");
    outputTripleVectorTo(out, vTreeCentroids);
    
    fprintf(out, "\n\nTree clusters:\n\n");
    outputTripleVectorTo(out, vTreeClusters);
    
    vector< vector<int> > vCentroidsForSearchTerms  = vTreeCentroids[vTreeCentroids.size() - 1];
    vector< vector<int> > vClustersForSearchTerms   = vTreeClusters[vTreeClusters.size() - 1];
    
    for (int i = ((int)(vTreeCentroids.size()) - 1); i >= 0; i--) {
        fprintf(out, "\n\nvCentroidsForSearchTerms:\n\n");
        outputDualVectorTo(out, vCentroidsForSearchTerms);
        
        fprintf(out, "\n\nvClustersForSearchTerms:\n\n");
        outputDualVectorTo(out, vClustersForSearchTerms);
        
        int indexOfVCentroidForSearchTerms = indexOfVectorCentroidsForSearchTerms(searchTerms, vCentroidsForSearchTerms);
        fprintf(out, "\n\nIndex of vector centroid for search terms:\n\n%d", indexOfVCentroidForSearchTerms);
        
        fprintf(out, "\n\nVector centroid for search terms:\n\n");
        vector<int> vectorCentroid = vCentroidsForSearchTerms[indexOfVCentroidForSearchTerms];
        outputVectorTo(out, vectorCentroid);
        
        if (i != 0) {
            fprintf(out, "\n\nVector clusters for search terms:\n\n");
            vector<int> vectorClusters = vClustersForSearchTerms[indexOfVCentroidForSearchTerms];
            outputVectorTo(out, vectorClusters);
            
            vCentroidsForSearchTerms    = vectorCentroidsForSearchTerms(vTreeCentroids, i-1, vectorClusters);
            vClustersForSearchTerms     = vectorClustersForSearchTerms(vTreeClusters, i-1, vectorClusters);
        }
        
        fprintf(out, "\n\n");
    }
    
    fclose(out);
    
	return 0;
}
