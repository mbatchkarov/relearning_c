#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>

#define K 3
#define ROWS 1000
#define COLS 40

void dist_to_centroids(gsl_matrix *data, gsl_matrix *centroids, gsl_matrix *dist, gsl_vector_uint *cluster_assignment);

gsl_rng *r; /* global generator */

void init_random() {
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
}

void generate_data(gsl_matrix *out) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            double val = gsl_rng_uniform_pos(r);
            // add some artificial offsets to one dimension to make sure there are 3 clear clusters
            if (j == 0 && i < ROWS / K) {
                val += 10;
            }
            if (j == 0 && i > 2 * ROWS / K) {
                val -= 10;
            }
            gsl_matrix_set(out, i, j, val);
        }
    }
}

// pick K random datapoints from the original data to be centroids- wikipedia says that's the Forgy algorithm
gsl_vector_uint *init_centroids(int k, gsl_matrix *m) {
    gsl_vector_uint *indices = gsl_vector_uint_alloc(k);
    for (int i = 0; i < k; i++) {
        // TODO may contain duplicates!!!
        gsl_vector_uint_set(indices, i, gsl_rng_uniform_int(r, m->size1));
    }

    // TODO hardcode a perfect initialization to see if the algo converges
    gsl_vector_uint_set(indices, 0, ROWS / 6);
    gsl_vector_uint_set(indices, 1, ROWS / 2);
    gsl_vector_uint_set(indices, 2, 5 * ROWS / 6);
    return indices;
}

void print_array(int *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%i ", data[i]);
    }
    printf("\n");
}

void print_vec_uint(gsl_vector_uint *data) {
    for (int i = 0; i < data->size; i++) {
        printf("%i ", gsl_vector_uint_get(data, i));
    }
    printf("\n");
}

void print_vec(gsl_vector *data) {
    for (int i = 0; i < data->size; i++) {
        printf("%f, ", gsl_vector_get(data, i));
    }
    printf("\n");
}

void print_mat(gsl_matrix *m) {
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            printf("%f,", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

typedef struct iter_result {
    gsl_matrix *centroids;
    gsl_vector_uint *cluster_assignments;
} iter_result;

void update(iter_result *state, gsl_matrix *data) {
    // find the nearest centroid for each data point (row)
    gsl_matrix *dist = gsl_matrix_alloc(ROWS, K);
    gsl_matrix_set_all(dist, -1);
    dist_to_centroids(data, state->centroids, dist, state->cluster_assignments);

    // update centroids -> TODO this can overflow really fast
    gsl_matrix *new_centroids = gsl_matrix_alloc(state->centroids->size1, state->centroids->size2);
    gsl_vector_uint *counts = gsl_vector_uint_alloc(state->centroids->size1); // num items per cluster
    for (int i = 0; i < data->size1; i++) {
        int cluster_id = gsl_vector_uint_get(state->cluster_assignments, i);
        gsl_vector_uint_set(counts, cluster_id, 1 + gsl_vector_uint_get(counts, cluster_id));
        gsl_vector this_centroid = gsl_matrix_row(new_centroids, cluster_id).vector;
        gsl_vector this_row = gsl_matrix_row(data, i).vector;
        gsl_vector_add(&this_centroid, &this_row);
    }

    for (int i = 0; i < new_centroids->size1; i++) {
        for (int j = 0; j < new_centroids->size2; j++) {
            gsl_matrix_set(new_centroids, i, j,
                           gsl_matrix_get(new_centroids, i, j) / gsl_vector_uint_get(counts, i));
        }
    }

    state->centroids = new_centroids;
}

double euclidean_dist(gsl_vector_view x, gsl_vector_view y) {
    // TODO this can prob be done faster with blas
    gsl_vector *res = gsl_vector_alloc(x.vector.size);
    gsl_vector_memcpy(res, &x.vector);

    // compute sum((a - b) ^ 2)- don't need sqrt because we only care about argmin
    gsl_vector_sub(res, &y.vector); // stores result in 1st param
    gsl_vector_mul(res, res);
    return gsl_vector_sum(res);
}

void dist_to_centroids(gsl_matrix *data, gsl_matrix *centroids, gsl_matrix *dist, gsl_vector_uint *cluster_assignment) {
    for (int i = 0; i < dist->size1; i++) // data
    {
        double closest_dist = DBL_MAX;
        for (int j = 0; j < dist->size2; j++) // centroids
        {
            double d = euclidean_dist(gsl_matrix_row(data, i), gsl_matrix_row(centroids, j));
            gsl_matrix_set(dist, i, j, d); // store dist to each centroid for debugging
            if (d < closest_dist) {
                closest_dist = d;
                gsl_vector_uint_set(cluster_assignment, i, j);  // store ID of the nearest cluster
            }
        }
    }
}

void get_data_at(gsl_matrix *data, gsl_vector_uint *centroid_idx, gsl_matrix *centroids) {
    for (int i = 0; i < K; i++) {
        // copy centroid indices
        gsl_vector_view row = gsl_matrix_row(data, gsl_vector_uint_get(centroid_idx, i));
//        print_vec(&row.vector);
        gsl_matrix_set_row(centroids, i, &row.vector);
    }
}

void write_state(iter_result state, gsl_matrix *m) {
    FILE *fp = fopen("data.txt", "w");
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            fprintf(fp, "%f,", gsl_matrix_get(m, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    fp = fopen("centroids.txt", "w");
    for (size_t i = 0; i < state.centroids->size1; i++) {
        for (size_t j = 0; j < state.centroids->size2; j++) {
            fprintf(fp, "%f,", gsl_matrix_get(state.centroids, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    fp = fopen("clusters.txt", "w");
    for (size_t i = 0; i < state.cluster_assignments->size; i++) {
        fprintf(fp, "%i,", gsl_vector_uint_get(state.cluster_assignments, i));
    }
    fclose(fp);
}

int main(void) {
    init_random();

    gsl_matrix *m = gsl_matrix_alloc(ROWS, COLS);
    generate_data(m);

    gsl_vector_uint *initial_centroid_idxs = init_centroids(K, m);

    for(int iter = 0; iter < 20; iter++) {
        gsl_vector_uint *cluster_assignment = gsl_vector_uint_alloc(ROWS);

        gsl_matrix *centroids = gsl_matrix_alloc(K, COLS);
        get_data_at(m, initial_centroid_idxs, centroids);

        iter_result state;
        state.centroids = centroids;
        state.cluster_assignments = cluster_assignment;

        for (int i = 0; i < 5; i++) // TODO or until convergence
        {
            update(&state, m);
//            print_vec_uint(state.cluster_assignments);
        }
    }
//    write_state(state, m);
    free(initial_centroid_idxs);
    return 0;
}
