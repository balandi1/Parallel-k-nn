
// global variables required
uint64_t id;
uint64_t n_points;
uint64_t n_dims;
uint64_t no_of_cores;
uint64_t q_id;
uint64_t n_queries;
uint64_t q_n_dims;
uint64_t n_neighbors;
char *resfilename;
uint64_t idCompare=0;

//class to store the nearest neighbors
class Neighbor
{
	public:
	float distance;
	uint64_t index;
	
};

// struct for comparing the neighbor distances with query node
struct LessThanByDist
{
  bool operator()(const Neighbor& lhs, const Neighbor& rhs) const
  {
    return lhs.distance < rhs.distance;
  }
};

// Kd tree Node structure 
struct KdNode
{
	uint64_t index;
	std::vector<float> x1;
	
	std::priority_queue<Neighbor, std::vector<Neighbor>,LessThanByDist> best_n;
	
    struct KdNode *left;
	struct KdNode *right;
	
	~KdNode()
	{
		delete left;
		delete right;
	}
	
};

//compare operator for comparing the node points at a particular dimension 
bool mycompare(KdNode lhs, KdNode rhs) { return lhs.x1[idCompare] < rhs.x1[idCompare]; }

// node pointers
struct KdNode* root;
struct KdNode* wp;
struct KdNode* qpoints;

//functions to perform operations on kd tree
inline double distance(struct KdNode *a, struct KdNode *b, int dimension);
void verifyTree(struct KdNode* root,int level);
struct KdNode* create_tree(struct KdNode *t, int len, int i, int dim);
void knn1(struct KdNode *root, struct KdNode *nd, int i, int dim, uint64_t k_neighbors);
