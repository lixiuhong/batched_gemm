#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<fstream>

#define random(x) (rand()%(x))

int main(int argc, char *argv[]){

	if (argc<3){
		printf("Usage: please input two integers\n");
		printf("The first one represents the largest matrix size (M, N)\n");
		printf("The second one represents the K\n");
		exit(EXIT_FAILURE);	
	}

	
	std::fstream fs;
	fs.open("../data/input");
	if (!fs.is_open()){
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}
	
	int e = atoi(argv[1]);
	int log_e = 0;

	while(e>=16){
		e = e>>1;
		log_e++;
	}	
	
	int K = atoi(argv[2]);
	//read matrix config	
	for (int i=0; i<256; ++i){
		int M = 16<<random(log_e);
		int N = 16<<random(log_e);
		fs<<M<<' '<<N<<' '<<K<<std::endl;
	}

	return EXIT_SUCCESS;	
}
