#include <iostream>
#include <fcntl.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>
#include<random>
#include<fstream>
#include<algorithm>
#include <chrono>
#include<thread>
#include<queue>
#include<future>
#include "KdNode.hpp"
#include "FileReader.hpp"

/* function to calculate euclidean distance on a dimension */
inline double distance(struct KdNode *a, struct KdNode *b, int dimension)
{
    double n, ans = 0;
    while (dimension--) {
        n = a->x1[dimension] - b->x1[dimension];
        ans += n * n;
    }
    return ans;
}

/* function to verify the tree construction */
void verifyTree(struct KdNode* root,int level)
{
	std::cout<<"Root for level "<<level<<"="<<root->x1[0]<<std::endl;
	if(root->left)
	{
		std::cout<<"Left for parent:"<<root->x1[0]<<" ";
		verifyTree(root->left,++level);
	}
	if(root->right)
	{
		std::cout<<"Right for parent:"<<root->x1[0]<<" ";
		verifyTree(root->right,++level);
	}
}

/* function to build k-d tree for given points and dimensions */
struct KdNode*
create_tree(struct KdNode *t, int len, int i, int dim)
{
	if (!len) return 0;
	
	idCompare=i;
	std::sort(t,t+len,mycompare);
    
	struct KdNode *middle = t + (len / 2);
	uint64_t actual_index=middle-wp;
	middle->index=actual_index;
	
	i = (i + 1) % dim;
	
	middle->left  = create_tree(t, len/2, i, dim);
	middle->right = create_tree(middle + 1,len - (len/2) - 1, i, dim);
	
    return middle;
}

 /* function to perform knn search to find k nearest neighbor for a query node */
 void knn1(struct KdNode *root, struct KdNode *nd, int i, int dim,
		uint64_t k_neighbors)
{
	double d, dx, dx2;
	
    if (!root) return;
	
	d = distance(root, nd, dim);
    dx = root->x1[i] - nd->x1[i];
    dx2 = dx * dx;
	
	if(nd->best_n.size()==0)
	{
		Neighbor n;
		n.distance=d;
		n.index=root->index;
		nd->best_n.push(n);
	}
	else if(d<nd->best_n.top().distance)
	{
		if(!(nd->best_n.size()<k_neighbors))
		{
			nd->best_n.pop();
			
		}
		
		Neighbor cn;
		cn.distance=d;
		cn.index=root->index;
		nd->best_n.push(cn);
	}
	else
	{
		if(nd->best_n.size()<k_neighbors)
		{
			Neighbor bn;
			bn.distance=d;
			bn.index=root->index;
			nd->best_n.push(bn);
		}
		else
			return;
	}
	
	if(++i >= dim) 
		i = 0;
	
	knn1(dx > 0 ? root->left : root->right, nd, i, dim, k_neighbors);
	knn1(dx > 0 ? root->right : root->left, nd, i, dim,k_neighbors);
	
}

/* function to generate a 64 bit result ID */ 
uint64_t generateResultID()
{
	std::uniform_int_distribution<uint64_t> d(std::llround(std::pow(2,61)), std::llround(std::pow(2,62)));
	std::random_device rd2("/dev/random");
	return d(rd2);
}

/* function to write result data in binary format to given file */
void writeResult()
{
	std::string type="RESULT";
	
	uint64_t resultID=generateResultID();
	char pad='\0';
	
	while(type.size()<8)
	{
		type+=pad;
	}
	std::ofstream file (resfilename, std::ios::binary);
	
	file<<type;
	file.write(reinterpret_cast<const char *>(&id), sizeof(id));
	file.write(reinterpret_cast<const char *>(&q_id), sizeof(q_id));
	file.write(reinterpret_cast<const char *>(&resultID), sizeof(resultID));
	file.write(reinterpret_cast<const char *>(&n_queries), sizeof(n_queries));
	file.write(reinterpret_cast<const char *>(&n_dims), sizeof(n_dims));
	file.write(reinterpret_cast<const char *>(&n_neighbors), sizeof(n_neighbors));	
	
	for(int i = 0; i < n_queries; i++) {
		
		while(!qpoints[i].best_n.empty())
		{
			Neighbor node=qpoints[i].best_n.top();
			qpoints[i].best_n.pop();
			
			for(int k=0;k<n_dims;k++)
			{
				float point=wp[node.index].x1[k];
				file.write(reinterpret_cast<const char *>(&point), sizeof(point));
			}
		}
	}
	file.close();
}

/* function to read training file and query file and store data in node structure 
   this function is used from the sample reader provided */ 
void readInputs(const std::string &fn) {

    int fd = open(fn.c_str(), O_RDONLY);
    if (fd < 0) {
        int en = errno;
        std::cerr << "Couldn't open " << fn << ": " << strerror(en) << "." << std::endl;
        exit(2);
    }
    // Get the actual size of the file.
    struct stat sb;
    int rv = fstat(fd, &sb); 
	assert(rv == 0);

    // Use some flags that will hopefully improve performance.
    void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
    }
    char *file_mem = (char *) vp;

    // Tell the kernel that it should evict the pages as soon as possible.
    rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); 
	assert(rv == 0);

    rv = close(fd); 
	assert(rv == 0);

    // Prefix to print before every line, to improve readability.
    std::string pref("    ");

    int n = strnlen(file_mem, 8);
    std::string file_type(file_mem, n);

    // Start to read data, skip the file type string.
    Reader reader{file_mem + 8};

    if (file_type == "TRAINING") 
	{
        reader >> id >> n_points >> n_dims;

		wp=(KdNode*)malloc(n_points*sizeof(KdNode));
        
		for (std::uint64_t i = 0; i < n_points; i++) {
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
				wp[i].x1.push_back(f);
            }
        }
    }
	else if (file_type == "QUERY") {
        reader >> q_id >> n_queries >> q_n_dims >> n_neighbors;
        
        qpoints=(KdNode*)malloc(n_queries*sizeof(KdNode));
		for (std::uint64_t i = 0; i < n_queries; i++) {
			qpoints[i].x1.reserve(q_n_dims);
			
			
            for (std::uint64_t j = 0; j < q_n_dims; j++) {
                float f;
                reader >> f;
				qpoints[i].x1.push_back(f);
            }
        }	

    }
    rv = munmap(file_mem, sb.st_size); 
	assert(rv == 0);
	//delete vp;
	//delete file_mem;
}


/* main function */
int main(int argc, char **argv) {

	// validation
	if(argc<5)
	{
		std::cout<<"Too few arguments provided"<<std::endl;
		exit(1);
	}
	
	no_of_cores=atoi(argv[1]);
	// get result file name
	resfilename=argv[4];
	
	// concurrently read training and query file
    std::thread t1(readInputs,argv[2]);
	std::thread t2(readInputs,argv[3]);
	
	t1.join();
	t2.join();
	
	 //build tree for training points
	auto bstart = std::chrono::high_resolution_clock::now(); 
	root = create_tree(wp, n_points, 0, n_dims);
	auto bstop = std::chrono::high_resolution_clock::now();
	
	auto build_time = std::chrono::duration_cast<std::chrono::seconds>(bstop - bstart); 
	std::cout<<"Time taken to build tree (in seconds): "<<build_time.count()<<std::endl;

	// knn search
	auto qstart = std::chrono::high_resolution_clock::now(); 
	
	// get the thread limit for the system
	const size_t n_threads = no_of_cores;
	{
		// maintain a vector of total allowed threads
		std::vector<std::thread> threads(n_threads);
		
		// divide the queries and assign fixed number of queries to each thread
		for(int t = 0;t<no_of_cores;t++)
		{
			threads[t] = std::thread(
			std::bind(
			[&](const int start, const int end, const int t)
			{
				for(int i = start;i<end;i++)
				{
					{
						knn1(root, &qpoints[i], 0, n_dims, n_neighbors);
					}
				}
			},t*n_queries/n_threads,(t+1)==n_threads?n_queries:(t+1)*n_queries/n_threads,t));
		}
	
		std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
	}
	auto qstop = std::chrono::high_resolution_clock::now(); 
	auto query_time = std::chrono::duration_cast<std::chrono::seconds>(qstop - qstart); 
	
	std::cout<<"Time taken to execute queries (in seconds): "<<query_time.count()<<std::endl;
	
	// write results to file in binary format
	writeResult();
	//delete wp;
	//delete qpoints;
	

}
