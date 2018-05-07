

void identity_matrix (double *, int);

double squared_norm  (double *, int, int);
double inner_product (double *, int, double *, int, int);

void matrix_product (double *, double *, double *, int, int, int);
double trace_of_product (double *, double *, int);

int cholesky (double *, int, double *);
int inverse_from_cholesky (double *, double *, double *, int);

void fill_lower_triangle (double *, int);
void fill_upper_triangle (double *, int);

void forward_solve (double *, double *, int, double *, int, int);
void backward_solve (double *, double *, int, double *, int, int);

int jacobi (double *, double *, double, int);


