/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

#include <math.h>
#include "superlu_ddefs.h"

char mtx_name_glo[100];
FILE* result_file;

void pdgssvx(superlu_dist_options_t *options, SuperMatrix *A,
			 dScalePermstruct_t *ScalePermstruct,
			 double B[], int ldb, int nrhs, gridinfo_t *grid,
			 dLUstruct_t *LUstruct, dSOLVEstruct_t *SOLVEstruct, double *berr,
			 SuperLUStat_t *stat, int *info)
{
	NRformat_loc *Astore;
	SuperMatrix GA; /* Global A in NC format */
	NCformat *GAstore;
	double *a_GA;
	SuperMatrix GAC; /* Global A in NCP format (add n end pointers) */
	NCPformat *GACstore;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	Glu_freeable_t *Glu_freeable; // (Yida) : Glu_freeable stores the struct after symbolic factorisation.
	fact_t Fact;
	double *a;
	int_t *colptr, *rowind;
	int_t *perm_r;			/* row permutations from partial pivoting */
	int_t *perm_c;			/* column permutation vector */
	int_t *etree;			/* elimination tree */
	int_t *rowptr, *colind; /* Local A in NR*/
	int_t nnz_loc, nnz;
	int m_loc, fst_row, icol, iinfo;
	int colequ, Equil, factored, job, notran, rowequ, need_value;
	int_t i, j, irow, m, n;
	int permc_spec;
	int iam, iam_g;
	int ldx; /* LDA for matrix X (local). */
	char equed[1], norm[1];
	double *C, *R, *C1, *R1, amax, anorm, colcnd, rowcnd;
	double *X, *b_col, *b_work, *x_col;
	double t;
	float GA_mem_use = 0.0;	  /* memory usage by global A */
	float dist_mem_use = 0.0; /* memory usage during distribution */
	superlu_dist_mem_usage_t num_mem_usage, symb_mem_usage;
	int64_t nnzLU;
	int_t nnz_tot;
	double *nzval_a;
	double asum, asum_tot, lsum, lsum_tot;
	int_t nsupers, nsupers_j;
	int_t lk, k, knsupc, nsupr;
	int_t *lsub, *xsup;
	double *lusup;
	LUstruct->dt = 'd';

	/* Structures needed for parallel symbolic factorization */
	int_t *sizes, *fstVtxSep, parSymbFact;
	int noDomains, nprocs_num;
	MPI_Comm symb_comm; /* communicator for symbolic factorization */
	int col, key;		/* parameters for creating a new communicator */
	Pslu_freeable_t Pslu_freeable;
	float flinfo;

	/* Initialization. */
	m = A->nrow;
	n = A->ncol;
	Astore = (NRformat_loc *)A->Store;
	nnz_loc = Astore->nnz_loc;
	m_loc = Astore->m_loc;
	fst_row = Astore->fst_row;
	a = (double *)Astore->nzval;
	rowptr = Astore->rowptr;
	colind = Astore->colind;
	sizes = NULL;
	fstVtxSep = NULL;
	symb_comm = MPI_COMM_NULL;
	num_mem_usage.for_lu = num_mem_usage.total = 0.0;
	symb_mem_usage.for_lu = symb_mem_usage.total = 0.0;

	/* Test the input parameters. */
	*info = 0;
	Fact = options->Fact;
	if (Fact < DOFACT || Fact > FACTORED)
		*info = -1;
	else if (options->RowPerm < NOROWPERM || options->RowPerm > MY_PERMR)
		*info = -1;
	else if (options->ColPerm < NATURAL || options->ColPerm > MY_PERMC)
		*info = -1;
	else if (options->IterRefine < NOREFINE || options->IterRefine > SLU_EXTRA)
		*info = -1;
	else if (options->IterRefine == SLU_EXTRA)
	{
		*info = -1;
		printf("ERROR: Extra precise iterative refinement yet to support.\n");
	}
	else if (A->nrow != A->ncol || A->nrow < 0 || A->Stype != SLU_NR_loc || A->Dtype != SLU_D || A->Mtype != SLU_GE)
		*info = -2;
	else if (ldb < m_loc)
		*info = -5;
	else if (nrhs < 0)
		*info = -6;
	if (sp_ienv_dist(2, options) > sp_ienv_dist(3, options))
	{
		*info = -1;
		printf("ERROR: Relaxation (SUPERLU_RELAX) cannot be larger than max. supernode size (SUPERLU_MAXSUP).\n"
			   "\t-> Check parameter setting in sp_ienv_dist.c to correct error.\n");
	}
	if (*info)
	{
		i = -(*info);
		pxerr_dist("pdgssvx", grid, -*info);
		return;
	}

	factored = (Fact == FACTORED);
	Equil = (!factored && options->Equil == YES);
	notran = (options->Trans == NOTRANS);
	parSymbFact = options->ParSymbFact;

	iam = grid->iam;
	job = 5;
	if (factored || (Fact == SamePattern_SameRowPerm && Equil))
	{
		rowequ = (ScalePermstruct->DiagScale == ROW) ||
				 (ScalePermstruct->DiagScale == BOTH);
		colequ = (ScalePermstruct->DiagScale == COL) ||
				 (ScalePermstruct->DiagScale == BOTH);
	}
	else
		rowequ = colequ = FALSE;

	/* The following arrays are replicated on all processes. */
	perm_r = ScalePermstruct->perm_r;
	perm_c = ScalePermstruct->perm_c;
	etree = LUstruct->etree;
	R = ScalePermstruct->R;
	C = ScalePermstruct->C;
	/********/

	/* Not factored & ask for equilibration */
	if (Equil && Fact != SamePattern_SameRowPerm)
	{
		/* Allocate storage if not done so before. */
		switch (ScalePermstruct->DiagScale)
		{
		case NOEQUIL:
			if (!(R = (double *)doubleMalloc_dist(m)))
				ABORT("Malloc fails for R[].");
			if (!(C = (double *)doubleMalloc_dist(n)))
				ABORT("Malloc fails for C[].");
			ScalePermstruct->R = R;
			ScalePermstruct->C = C;
			break;
		case ROW:
			if (!(C = (double *)doubleMalloc_dist(n)))
				ABORT("Malloc fails for C[].");
			ScalePermstruct->C = C;
			break;
		case COL:
			if (!(R = (double *)doubleMalloc_dist(m)))
				ABORT("Malloc fails for R[].");
			ScalePermstruct->R = R;
			break;
		default:
			break;
		}
	}

	/* ------------------------------------------------------------
	 * Diagonal scaling to equilibrate the matrix. (simple scheme)
	 *   for row i = 1:n,  A(i,:) <- A(i,:) / max(abs(A(i,:));
	 *   for column j = 1:n,  A(:,j) <- A(:, j) / max(abs(A(:,j))
	 * ------------------------------------------------------------*/
	if (Equil)
	{
		t = SuperLU_timer_();

		if (Fact == SamePattern_SameRowPerm)
		{
			/* Reuse R and C. */
			switch (ScalePermstruct->DiagScale)
			{
			case NOEQUIL:
				break;
			case ROW:
				irow = fst_row;
				for (j = 0; j < m_loc; ++j)
				{
					for (i = rowptr[j]; i < rowptr[j + 1]; ++i)
					{
						a[i] *= R[irow]; /* Scale rows. */
					}
					++irow;
				}
				break;
			case COL:
				for (j = 0; j < m_loc; ++j)
					for (i = rowptr[j]; i < rowptr[j + 1]; ++i)
					{
						icol = colind[i];
						a[i] *= C[icol]; /* Scale columns. */
					}
				break;
			case BOTH:
				irow = fst_row;
				for (j = 0; j < m_loc; ++j)
				{
					for (i = rowptr[j]; i < rowptr[j + 1]; ++i)
					{
						icol = colind[i];
						a[i] *= R[irow] * C[icol]; /* Scale rows and cols. */
					}
					++irow;
				}
				break;
			}
		}
		else
		{ /* Compute R & C from scratch */
			/* Compute the row and column scalings. */
			pdgsequ(A, R, C, &rowcnd, &colcnd, &amax, &iinfo, grid);

			if (iinfo > 0)
			{
				if (iinfo <= m)
				{
					fprintf(stderr, "The %d-th row of A is exactly zero\n", (int)iinfo);
				}
				else
				{
					fprintf(stderr, "The %d-th column of A is exactly zero\n", (int)(iinfo - n));
				}
			}
			else if (iinfo < 0)
				return;

			/* Now iinfo == 0 */

			/* Equilibrate matrix A if it is badly-scaled.
			   A <-- diag(R)*A*diag(C)                     */
			pdlaqgs(A, R, C, rowcnd, colcnd, amax, equed);

			if (strncmp(equed, "R", 1) == 0)
			{
				ScalePermstruct->DiagScale = ROW;
				rowequ = ROW;
			}
			else if (strncmp(equed, "C", 1) == 0)
			{
				ScalePermstruct->DiagScale = COL;
				colequ = COL;
			}
			else if (strncmp(equed, "B", 1) == 0)
			{
				ScalePermstruct->DiagScale = BOTH;
				rowequ = ROW;
				colequ = COL;
			}
			else
				ScalePermstruct->DiagScale = NOEQUIL;
		} /* end if Fact ... */

		stat->utime[EQUIL] = SuperLU_timer_() - t;
	} /* end if Equil ... LAPACK style, not involving MC64 */

	if (!factored)
	{ /* Skip this if already factored. */
		/*
		 * For serial symbolic factorization, gather A from the distributed
		 * compressed row format to global A in compressed column format.
		 * Numerical values are gathered only when a row permutation
		 * for large diagonal is sought after.
		 */
		if (Fact != SamePattern_SameRowPerm && (parSymbFact == NO || options->RowPerm != NO))
		{
			/* Performs serial symbolic factorzation and/or MC64 */

			need_value = (options->RowPerm == LargeDiag_MC64);

			pdCompRow_loc_to_CompCol_global(need_value, A, grid, &GA);

			GAstore = (NCformat *)GA.Store;
			colptr = GAstore->colptr;
			rowind = GAstore->rowind;
			nnz = GAstore->nnz;
			GA_mem_use = (nnz + n + 1) * sizeof(int_t);

			if (need_value)
			{
				a_GA = (double *)GAstore->nzval;
				GA_mem_use += nnz * sizeof(double);
			}
			else
				assert(GAstore->nzval == NULL);
		}

		/* ------------------------------------------------------------
		   Find the row permutation Pr for A, and apply Pr*[GA].
	   GA is overwritten by Pr*[GA].
		   ------------------------------------------------------------*/
		if (options->RowPerm != NO)
		{
			t = SuperLU_timer_();
			if (Fact != SamePattern_SameRowPerm)
			{
				if (options->RowPerm == MY_PERMR)
				{ /* Use user's perm_r. */
					/* Permute the global matrix GA for symbfact() */
					for (i = 0; i < colptr[n]; ++i)
					{
						irow = rowind[i];
						rowind[i] = perm_r[irow];
					}
				}
				else if (options->RowPerm == LargeDiag_MC64)
				{
					/* Get a new perm_r[] from MC64 */
					if (job == 5)
					{
						/* Allocate storage for scaling factors. */
						if (!(R1 = doubleMalloc_dist(m)))
							ABORT("SUPERLU_MALLOC fails for R1[]");
						if (!(C1 = doubleMalloc_dist(n)))
							ABORT("SUPERLU_MALLOC fails for C1[]");
					}

					if (!iam)
					{ /* Process 0 finds a row permutation */
						iinfo = dldperm_dist(job, m, nnz, colptr, rowind, a_GA,
											 perm_r, R1, C1);

						MPI_Bcast(&iinfo, 1, MPI_INT, 0, grid->comm);
						if (iinfo == 0)
						{
							MPI_Bcast(perm_r, m, mpi_int_t, 0, grid->comm);
							if (job == 5 && Equil)
							{
								MPI_Bcast(R1, m, MPI_DOUBLE, 0, grid->comm);
								MPI_Bcast(C1, n, MPI_DOUBLE, 0, grid->comm);
							}
						}
					}
					else
					{
						MPI_Bcast(&iinfo, 1, MPI_INT, 0, grid->comm);
						if (iinfo == 0)
						{
							MPI_Bcast(perm_r, m, mpi_int_t, 0, grid->comm);
							if (job == 5 && Equil)
							{
								MPI_Bcast(R1, m, MPI_DOUBLE, 0, grid->comm);
								MPI_Bcast(C1, n, MPI_DOUBLE, 0, grid->comm);
							}
						}
					}

					if (iinfo && job == 5)
					{ /* Error return */
						SUPERLU_FREE(R1);
						SUPERLU_FREE(C1);
					}
					if (iinfo == 0) // (Yida) : iinfo == 0 means the matrix isn't a singular matrix.
					{
						if (job == 5)
						{
							if (Equil)
							{
								for (i = 0; i < n; ++i)
								{
									R1[i] = exp(R1[i]);
									C1[i] = exp(C1[i]);
								}

								/* Scale the distributed matrix further.
							   A <-- diag(R1)*A*diag(C1)            */
								irow = fst_row;
								for (j = 0; j < m_loc; ++j)
								{
									for (i = rowptr[j]; i < rowptr[j + 1]; ++i)
									{
										icol = colind[i];
										a[i] *= R1[irow] * C1[icol];
									}
									++irow;
								}

								/* Multiply together the scaling factors --
							   R/C from simple scheme, R1/C1 from MC64. */
								if (rowequ)
									for (i = 0; i < m; ++i)
										R[i] *= R1[i];
								else
									for (i = 0; i < m; ++i)
										R[i] = R1[i];
								if (colequ)
									for (i = 0; i < n; ++i)
										C[i] *= C1[i];
								else
									for (i = 0; i < n; ++i)
										C[i] = C1[i];

								ScalePermstruct->DiagScale = BOTH;
								rowequ = colequ = 1;

							} /* end Equil */

							/* Now permute global GA to prepare for symbfact() */
							for (j = 0; j < n; ++j)
							{
								for (i = colptr[j]; i < colptr[j + 1]; ++i)
								{
									irow = rowind[i];
									rowind[i] = perm_r[irow];
								}
							}
							SUPERLU_FREE(R1);
							SUPERLU_FREE(C1);
						}
						else
						{ /* job = 2,3,4 */
							for (j = 0; j < n; ++j)
							{
								for (i = colptr[j]; i < colptr[j + 1]; ++i)
								{
									irow = rowind[i];
									rowind[i] = perm_r[irow];
								} /* end for i ... */
							} /* end for j ... */
						} /* end else job ... */
					}
					else
					{ /* if iinfo != 0 */
						for (i = 0; i < m; ++i)
							perm_r[i] = i;
					}
				}
				else
				{ /* use LargeDiag_HWPM */
					if (iam == 0)
					{
						printf("CombBLAS is not available\n");
						fflush(stdout);
					}
				} /* end if options->RowPerm ... */

				t = SuperLU_timer_() - t;
				stat->utime[ROWPERM] = t;
			} /* end if Fact ... */
		}
		else
		{ /* options->RowPerm == NOROWPERM / NATURAL */
			for (i = 0; i < m; ++i)
				perm_r[i] = i;
		}
	} /* end if (!factored) */

	if (!factored || options->IterRefine)
	{
		/* Compute norm(A), which will be used to adjust small diagonal. */
		if (notran)
			*(unsigned char *)norm = '1';
		else
			*(unsigned char *)norm = 'I';
		anorm = pdlangs(norm, A, grid);
	}

	/* ------------------------------------------------------------
	   Perform the LU factorization: symbolic factorization,
	   redistribution, and numerical factorization.
	   ------------------------------------------------------------*/
	if (!factored)
	{
		t = SuperLU_timer_();
		/*
		 * Get column permutation vector perm_c[], according to permc_spec:
		 *   permc_spec = NATURAL:  natural ordering
		 *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
		 *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
		 *   permc_spec = METIS_AT_PLUS_A: METIS on structure of A'+A
		 *   permc_spec = PARMETIS: parallel METIS on structure of A'+A
		 *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
		 */
		permc_spec = options->ColPerm;

		if (parSymbFact == YES || permc_spec == PARMETIS)
		{
			nprocs_num = grid->nprow * grid->npcol;
			noDomains = (int)(pow(2, ((int)LOG2(nprocs_num))));

			/* create a new communicator for the first noDomains
				   processes in grid->comm */
			key = iam;
			if (iam < noDomains)
				col = 0;
			else
				col = MPI_UNDEFINED;
			MPI_Comm_split(grid->comm, col, key, &symb_comm);

			if (permc_spec == NATURAL || permc_spec == MY_PERMC)
			{
				if (permc_spec == NATURAL)
				{
					for (j = 0; j < n; ++j)
						perm_c[j] = j;
				}
				if (!(sizes = intMalloc_dist(2 * noDomains)))
					ABORT("SUPERLU_MALLOC fails for sizes.");
				if (!(fstVtxSep = intMalloc_dist(2 * noDomains)))
					ABORT("SUPERLU_MALLOC fails for fstVtxSep.");
				for (i = 0; i < 2 * noDomains - 2; ++i)
				{
					sizes[i] = 0;
					fstVtxSep[i] = 0;
				}
				sizes[2 * noDomains - 2] = m;
				fstVtxSep[2 * noDomains - 2] = 0;
			}
			else if (permc_spec != PARMETIS)
			{ /* same as before */
				printf("{" IFMT "," IFMT "}: pdgssvx: invalid ColPerm option when ParSymbfact is used\n",
					   MYROW(grid->iam, grid), MYCOL(grid->iam, grid));
			}
		} /* end preparing for parallel symbolic */

		if (permc_spec != MY_PERMC && Fact == DOFACT)
		{
			/* Reuse perm_c if Fact == SamePattern, or SamePattern_SameRowPerm */
			if (permc_spec == PARMETIS)
			{
				/* Get column permutation vector in perm_c.                    *
				 * This routine takes as input the distributed input matrix A  *
				 * and does not modify it.  It also allocates memory for       *
				 * sizes[] and fstVtxSep[] arrays, that contain information    *
				 * on the separator tree computed by ParMETIS.                 */
				flinfo = get_perm_c_parmetis(A, perm_r, perm_c, nprocs_num,
											 noDomains, &sizes, &fstVtxSep,
											 grid, &symb_comm); // (Yida) : Call ParMETIS_V3_NodeND
				if (flinfo > 0)
				{
					fprintf(stderr, "Insufficient memory for get_perm_c parmetis\n");
					*info = flinfo;
					return;
				}
			}
			else
			{
				get_perm_c_dist(iam, permc_spec, &GA, perm_c);
			}
		}

		stat->utime[COLPERM] = SuperLU_timer_() - t;

		/* Symbolic factorization. */
		if (Fact != SamePattern_SameRowPerm)
		{
			if (parSymbFact == NO)
			{ /* Perform serial symbolic factorization */
				/* GA = Pr*A, perm_r[] is already applied. */
				int_t *GACcolbeg, *GACcolend, *GACrowind;

				/* Compute the elimination tree of Pc*(A^T+A)*Pc^T or Pc*A^T*A*Pc^T
				   (a.k.a. column etree), depending on the choice of ColPerm.
				   Adjust perm_c[] to be consistent with a postorder of etree.
				   Permute columns of A to form A*Pc'.
			 	   After this routine, GAC = GA*Pc^T.  */
				sp_colorder(options, &GA, perm_c, etree, &GAC);

				/* Form Pc*A*Pc^T to preserve the diagonal of the matrix GAC. */
				GACstore = (NCPformat *)GAC.Store;
				GACcolbeg = GACstore->colbeg;
				GACcolend = GACstore->colend;
				GACrowind = GACstore->rowind;
				for (j = 0; j < n; ++j)
				{
					for (i = GACcolbeg[j]; i < GACcolend[j]; ++i)
					{
						irow = GACrowind[i];
						GACrowind[i] = perm_c[irow];
					}
				}

				/* Perform a symbolic factorization on Pc*Pr*A*Pc^T and set up
					   the nonzero data structures for L & U. */
				t = SuperLU_timer_();
				if (!(Glu_freeable = (Glu_freeable_t *)
						  SUPERLU_MALLOC(sizeof(Glu_freeable_t))))
					ABORT("Malloc fails for Glu_freeable.");

				/* Every process does this.
			   returned value (-iinfo) is the size of lsub[], incuding pruned graph.*/
				int_t linfo;
				linfo = symbfact(options, iam, &GAC, perm_c, etree,
								 Glu_persist, Glu_freeable);
				nnzLU = Glu_freeable->nnzLU;
				stat->utime[SYMBFAC] = SuperLU_timer_() - t;
				if (linfo <= 0)
				{ /* Successful return */
					QuerySpace_dist(n, -linfo, Glu_freeable, &symb_mem_usage);
				}
				else
				{ /* symbfact out of memory */
					if (!iam)
						fprintf(stderr, "symbfact() error returns " IFMT "\n", linfo);
					*info = linfo;
					return;
				}
			} /* end serial symbolic factorization */
			else
			{ /* parallel symbolic factorization */
				t = SuperLU_timer_();
				flinfo = symbfact_dist(options, nprocs_num, noDomains,
									   A, perm_c, perm_r,
									   sizes, fstVtxSep, &Pslu_freeable,
									   &(grid->comm), &symb_comm,
									   &symb_mem_usage);
				nnzLU = Pslu_freeable.nnzLU;
				stat->utime[SYMBFAC] = SuperLU_timer_() - t;
				if (flinfo > 0)
				{
					fprintf(stderr, "Insufficient memory for parallel symbolic factorization.");
					*info = flinfo;
					return;
				}
			}

			/* Destroy global GA */
			if (parSymbFact == NO || options->RowPerm != NO)
				Destroy_CompCol_Matrix_dist(&GA);
			if (parSymbFact == NO)
				Destroy_CompCol_Permuted_dist(&GAC);

		} /* end if Fact != SamePattern_SameRowPerm ... */

		if (sizes)
			SUPERLU_FREE(sizes);
		if (fstVtxSep)
			SUPERLU_FREE(fstVtxSep);
		if (symb_comm != MPI_COMM_NULL)
			MPI_Comm_free(&symb_comm);

		/* Distribute entries of A into L & U data structures. */
		if (parSymbFact == NO)
		{
			/* CASE OF SERIAL SYMBOLIC */
			/* Apply column permutation to the original distributed A */
			for (j = 0; j < nnz_loc; ++j)
				colind[j] = perm_c[colind[j]];

			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc^T into L and U storage.
			   NOTE: the row permutation Pc*Pr is applied internally in the
			   distribution routine. */
			t = SuperLU_timer_();
			dist_mem_use = pddistribute(options, n, A, ScalePermstruct,
										Glu_freeable, LUstruct, grid);
			stat->utime[DIST] = SuperLU_timer_() - t;

			/* Deallocate storage used in symbolic factorization. */
			if (Fact != SamePattern_SameRowPerm)
			{
				iinfo = symbfact_SubFree(Glu_freeable);
				SUPERLU_FREE(Glu_freeable);
			}
		}
		else
		{ /* CASE OF PARALLEL SYMBOLIC */
			/* Distribute Pc*Pr*diag(R)*A*diag(C)*Pc' into L and U storage.
			   NOTE: the row permutation Pc*Pr is applied internally in the
			   distribution routine. */
			/* Apply column permutation to the original distributed A */
			for (j = 0; j < nnz_loc; ++j)
				colind[j] = perm_c[colind[j]];

			t = SuperLU_timer_();
			dist_mem_use = ddist_psymbtonum(options, n, A, ScalePermstruct,
											&Pslu_freeable, LUstruct, grid);

			/* dist_mem_use = memDist + memNLU */
			if (dist_mem_use > 0)
				ABORT("Not enough memory available for dist_psymbtonum\n");

			stat->utime[DIST] = SuperLU_timer_() - t;
		}

		/* Flatten L metadata into one buffer. */
		if (Fact != SamePattern_SameRowPerm)
		{
			pdflatten_LDATA(options, n, LUstruct, grid, stat);
		}

		int cuda_device = iam;
		int cuda_dev_cnt;
		cudaGetDeviceCount(&cuda_dev_cnt);
		cudaSetDevice(cuda_device%cuda_dev_cnt);

		xsup = Glu_persist->xsup;
		nsupers = Glu_persist->supno[n - 1] + 1;
		for(int brow=0;brow<nsupers;brow++){
			if(iam / grid->npcol == brow % grid->nprow){
				int_t* metau = LUstruct->Llu->Ufstnz_br_ptr[LBi(brow,  grid)];
				double* old_value = LUstruct->Llu->Unzval_br_ptr[LBi(brow,  grid)];
				if(metau){
					int u_metaidx = 3;
					int u_iblk = 0;
					int u_nblk = metau[0];
					int nrow = SuperSize(brow);
					int ncol_effective = 0;
					while(u_iblk < u_nblk){
						int ncol_math = SuperSize(metau[u_metaidx]);
						int ncol_this_tile = 0;
						for(int col = 0; col < ncol_math; col++){
							if(metau[u_metaidx+2+col] != FstBlockC(brow+1)){
								// metau[u_metaidx+2+col] = FstBlockC(metau[u_metaidx]);
								ncol_effective++;
								ncol_this_tile++;
							}
						}
						metau[u_metaidx + 1] = ncol_this_tile * nrow;
						u_metaidx += (2 + SuperSize(metau[u_metaidx+0]));
						u_iblk++;
					}

					double* new_value = (double*)malloc(sizeof(double) * ncol_effective * nrow);
					if(new_value == NULL){
						printf("[WARNING] Malloc failed.\n");
					}
					memset(new_value, 0, sizeof(double) * ncol_effective * nrow);

					metau[1] = ncol_effective * nrow;

					u_metaidx = 3;
					u_iblk = 0;
					unsigned long long oldval_offset = 0;
					unsigned long long newval_offset = 0;
					while(u_iblk < u_nblk){
						int ncol_math = SuperSize(metau[u_metaidx]);
						for(int col = 0; col < ncol_math; col++){
							if(metau[u_metaidx+2+col] != FstBlockC(brow+1)){
								int incol_offset = metau[u_metaidx+2+col] - FstBlockC(brow);
								// printf("incol_offset = %d  %d %d %d\n", incol_offset, 
								// 	metau[u_metaidx+2+col], FstBlockC(brow), brow);
								for(int row = incol_offset; row < nrow; row++){
									new_value[newval_offset + row] = old_value[oldval_offset + row - incol_offset];
								}
								metau[u_metaidx+2+col] = FstBlockC(brow);
								newval_offset += nrow;
								oldval_offset += (nrow - incol_offset);
								if(newval_offset > (ncol_effective * nrow)){
									printf("[WARNING] %d > %d\n", newval_offset, ncol_effective * nrow);
								}
							}
						}
						u_metaidx += (2 + SuperSize(metau[u_metaidx]));
						u_iblk++;
					}
					LUstruct->Llu->Unzval_br_ptr[LBi(brow,  grid)] = new_value;
					// SUPERLU_FREE(old_value);
				}
			}
		}

		dLocalLU_t* Llu = LUstruct->Llu;
		int_t **Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
		double **Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
		int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
		double **Unzval_br_ptr = Llu->Unzval_br_ptr;



		Llu->l_brow_idx_hh = malloc(sizeof(int_t*)*nsupers);
		Llu->l_tile_offset_hh = malloc(sizeof(int_t*)*nsupers);
		Llu->l_bcol_localperm_hh = malloc(sizeof(int_t*)*nsupers);

		Llu->l_nblk_bcol_prefixsum_h = malloc(sizeof(int_t) * (nsupers+1));
		Llu->l_brow_idx_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->l_tile_offset_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->l_bcol_localperm_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->l_bcol_val_hd = malloc(sizeof(double*)*nsupers);

		gpuMalloc(&(Llu->l_nblk_bcol_prefixsum_d), sizeof(int_t*)*(nsupers+1));
		gpuMalloc(&(Llu->l_brow_idx_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->l_tile_offset_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->l_bcol_localperm_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->l_bcol_val_dd), sizeof(double*)*nsupers);

		Llu->l_nblk_bcol_prefixsum_h[0] = 0;
		for (int_t bcol = 0; bcol < nsupers; bcol++)
        {
			int_t nsupc = SuperSize(bcol);

			Llu->l_nblk_bcol_prefixsum_h[bcol+1] = Llu->l_nblk_bcol_prefixsum_h[bcol];

            if(bcol % grid->npcol != iam % grid->npcol){
                continue;
            }
            int_t *metaa = Lrowind_bc_ptr[LBj(bcol, grid)];
            if(metaa == NULL){
                continue;
            }


            int_t l_metaidx = 2;
            int_t l_blkidx = 0;
            double *valuea = Lnzval_bc_ptr[LBj(bcol, grid)];
			int_t valuea_idx = 0;
            int_t a_metaidx = 2;
            int_t a_blkidx = 0;
            int_t a_nblk = metaa[0];
            int_t a_lda = metaa[1];

			Llu->l_nblk_bcol_prefixsum_h[bcol+1] = Llu->l_nblk_bcol_prefixsum_h[bcol] + a_nblk;

			Llu->l_brow_idx_hh[bcol] = malloc(sizeof(int_t)*a_nblk);
			Llu->l_tile_offset_hh[bcol] = malloc(sizeof(int_t)*(a_nblk+1));
			Llu->l_bcol_localperm_hh[bcol] = malloc(sizeof(int_t)*a_lda);

			gpuMalloc(&(Llu->l_brow_idx_hd[bcol]), sizeof(int_t)*a_nblk);
			gpuMalloc(&(Llu->l_tile_offset_hd[bcol]), sizeof(int_t)*(a_nblk+1));
			gpuMalloc(&(Llu->l_bcol_localperm_hd[bcol]), sizeof(int_t)*a_lda);
			gpuMalloc(&(Llu->l_bcol_val_hd[bcol]), sizeof(double)*nsupc*a_lda);

            while (a_blkidx < a_nblk)
            {
				int_t row = metaa[a_metaidx + 0];
				int_t dest_nbrow = metaa[a_metaidx + 1];

				Llu->l_brow_idx_hh[bcol][a_blkidx] = row;
				Llu->l_tile_offset_hh[bcol][a_blkidx] = valuea_idx;
				// memcpy(&(Llu->l_bcol_localperm_hh[bcol][valuea_idx]), &(metaa[a_metaidx+2]), sizeof(int_t)*dest_nbrow);
				for(int i=0;i<dest_nbrow;i++){
					Llu->l_bcol_localperm_hh[bcol][valuea_idx+i] = metaa[a_metaidx+2+i] - FstBlockC(metaa[a_metaidx + 0]);
				}

				valuea_idx += metaa[a_metaidx + 1];
				a_blkidx++;
				a_metaidx += (2 + metaa[a_metaidx + 1]);
            }
			Llu->l_tile_offset_hh[bcol][a_nblk] = valuea_idx;

			gpuMemcpy(Llu->l_brow_idx_hd[bcol], Llu->l_brow_idx_hh[bcol], sizeof(int_t)*a_nblk, gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->l_tile_offset_hd[bcol], Llu->l_tile_offset_hh[bcol], sizeof(int_t)*(a_nblk+1), gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->l_bcol_localperm_hd[bcol], Llu->l_bcol_localperm_hh[bcol], sizeof(int_t)*a_lda, gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->l_bcol_val_hd[bcol], valuea, sizeof(double)*nsupc*a_lda, gpuMemcpyHostToDevice);
	

			int errid = 0;
			if ((errid = gpuDeviceSynchronize()) != cudaSuccess)
			{
				printf("cuda error %d %s:%d\n", errid, __FILE__, __LINE__);
				exit(1);
			}
		}

		gpuMemcpy(Llu->l_nblk_bcol_prefixsum_d, Llu->l_nblk_bcol_prefixsum_h, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->l_brow_idx_dd, Llu->l_brow_idx_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->l_tile_offset_dd, Llu->l_tile_offset_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->l_bcol_localperm_dd, Llu->l_bcol_localperm_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->l_bcol_val_dd, Llu->l_bcol_val_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);





		Llu->u_bcol_idx_hh = malloc(sizeof(int_t*)*nsupers);
		Llu->u_tile_offset_hh = malloc(sizeof(int_t*)*nsupers);
		Llu->u_brow_localperm_hh = malloc(sizeof(int_t*)*nsupers);

		Llu->u_nblk_brow_prefixsum_h = malloc(sizeof(int_t) * (nsupers+1));
		Llu->u_bcol_idx_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->u_tile_offset_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->u_brow_localperm_hd = malloc(sizeof(int_t*)*nsupers);
		Llu->u_brow_val_hd = malloc(sizeof(double*)*nsupers);

		gpuMalloc(&(Llu->u_nblk_brow_prefixsum_d), sizeof(int_t*)*(nsupers+1));
		gpuMalloc(&(Llu->u_bcol_idx_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->u_tile_offset_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->u_brow_localperm_dd), sizeof(int_t*)*nsupers);
		gpuMalloc(&(Llu->u_brow_val_dd), sizeof(double*)*nsupers);

		Llu->u_nblk_brow_prefixsum_h[0] = 0;
		for (int_t brow = 0; brow < nsupers; brow++)
        {
			int_t nsupc = SuperSize(brow);

			Llu->u_nblk_brow_prefixsum_h[brow+1] = Llu->u_nblk_brow_prefixsum_h[brow];

            if(brow % grid->nprow != iam / grid->npcol){
                continue;
            }
            int_t *metaa = Ufstnz_br_ptr[LBi(brow, grid)];
            if(metaa == NULL){
                continue;
            }

			int_t u_nblk = metaa[0];
			int_t u_lda = nsupc;

            double *valuea = Unzval_br_ptr[LBi(brow, grid)];
            int_t a_metaidx = 3;
            int_t a_blkidx = 0;
            int_t a_nblk = metaa[0];
            int_t a_lda = SuperSize(brow);
            int_t l_a_fstr = FstBlockC(brow);
			int_t valuea_idx = 0;

			Llu->u_nblk_brow_prefixsum_h[brow+1] = Llu->u_nblk_brow_prefixsum_h[brow] + a_nblk;

			Llu->u_bcol_idx_hh[brow] = malloc(sizeof(int_t)*a_nblk);
			Llu->u_tile_offset_hh[brow] = malloc(sizeof(int_t)*(a_nblk+1));
			Llu->u_brow_localperm_hh[brow] = malloc(sizeof(int_t)*(metaa[1]/nsupc));
			memset(Llu->u_brow_localperm_hh[brow], 0, sizeof(int_t)*(metaa[1]/nsupc));

			gpuMalloc(&(Llu->u_bcol_idx_hd[brow]), sizeof(int_t)*a_nblk);
			gpuMalloc(&(Llu->u_tile_offset_hd[brow]), sizeof(int_t)*(a_nblk+1));
			gpuMalloc(&(Llu->u_brow_localperm_hd[brow]), sizeof(int_t)*(metaa[1]/nsupc));
			gpuMalloc(&(Llu->u_brow_val_hd[brow]), sizeof(double)*nsupc*(metaa[1]/nsupc));

            while (a_blkidx < a_nblk)
            {
				int_t bcol = metaa[a_metaidx + 0];
				int_t dest_ncol = metaa[a_metaidx + 1]/nsupc;

				Llu->u_bcol_idx_hh[brow][a_blkidx] = bcol;
				Llu->u_tile_offset_hh[brow][a_blkidx] = valuea_idx;
				int_t idx = 0;
				for(int_t col=0;col<SuperSize(bcol);col++){
					if(FstBlockC(brow + 1) != metaa[a_metaidx + 2 + col]){
						Llu->u_brow_localperm_hh[brow][valuea_idx+idx] = col;
						idx++;
					}
				}

				valuea_idx += metaa[a_metaidx + 1]/nsupc;
				a_blkidx++;
				a_metaidx += (2 + SuperSize(metaa[a_metaidx + 0]));
            }
			Llu->u_tile_offset_hh[brow][a_nblk] = valuea_idx;

			gpuMemcpy(Llu->u_bcol_idx_hd[brow], Llu->u_bcol_idx_hh[brow], sizeof(int_t)*a_nblk, gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->u_tile_offset_hd[brow], Llu->u_tile_offset_hh[brow], sizeof(int_t)*(a_nblk+1), gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->u_brow_localperm_hd[brow], Llu->u_brow_localperm_hh[brow], sizeof(int_t)*(metaa[1]/nsupc), gpuMemcpyHostToDevice);
			gpuMemcpy(Llu->u_brow_val_hd[brow], valuea, sizeof(double)*nsupc*(metaa[1]/nsupc), gpuMemcpyHostToDevice);

			int errid = 0;
			if ((errid = gpuDeviceSynchronize()) != cudaSuccess)
			{
				printf("cuda error %d %s:%d\n", errid, __FILE__, __LINE__);
				exit(1);
			}
		}

		gpuMemcpy(Llu->u_nblk_brow_prefixsum_d, Llu->u_nblk_brow_prefixsum_h, sizeof(int_t)*(nsupers+1), gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->u_bcol_idx_dd, Llu->u_bcol_idx_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->u_tile_offset_dd, Llu->u_tile_offset_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->u_brow_localperm_dd, Llu->u_brow_localperm_hd, sizeof(int_t*)*nsupers, gpuMemcpyHostToDevice);
		gpuMemcpy(Llu->u_brow_val_dd, Llu->u_brow_val_hd, sizeof(double*)*nsupers, gpuMemcpyHostToDevice);

		int errid = 0;
		if ((errid = gpuDeviceSynchronize()) != cudaSuccess)
		{
			printf("cuda error %d %s:%d\n", errid, __FILE__, __LINE__);
			exit(1);
		}

		/* Perform numerical factorization in parallel. */
		t = SuperLU_timer_();
		pdgstrf(options, m, n, anorm, LUstruct, grid, stat, info);
		stat->utime[FACT] = SuperLU_timer_() - t;

		if(!iam){
			fprintf(result_file, "%s, %lf\n", mtx_name_glo, stat->utime[FACT]);
		}

		// printf("%.2lf + %.2lf + %.2lf = %.2lf\n", stat->utime[FACT] - stat->time_tstrf_update, stat->time_tstrf_update/2, stat->time_tstrf_update/2, stat->utime[FACT]);

		if (options->PrintStat)
		{
			int_t TinyPivots;
			float for_lu, total, avg, loc_max;
			float mem_stage[3];
			struct
			{
				float val;
				int rank;
			} local_struct, global_struct;

			MPI_Reduce(&stat->TinyPivots, &TinyPivots, 1, mpi_int_t,
					   MPI_SUM, 0, grid->comm);
			stat->TinyPivots = TinyPivots;

			if (iam == 0)
			{
				printf("\n** Memory Usage **********************************\n");
			}

			/* Compute numerical factorization memeory */
			dQuerySpace_dist(n, LUstruct, grid, stat, &num_mem_usage);

			/*-- Compute high watermark of all stages --*/
			if (parSymbFact == TRUE)
			{
				/* The memory used in the redistribution routine
			   includes the memory used for storing the symbolic
			   structure and the memory allocated for numerical
			   factorization */
				/* parallel symbfact step:
				   (-flinfo) is the allocMem returned from symbfact_dist() */
				mem_stage[0] = symb_mem_usage.total + (-flinfo);

				/* see leading comment of dist_symbLU() */
				/* dist_mem_use = (memDist + memNLU) in ddist_psymbtonum() */
				mem_stage[1] = symb_mem_usage.for_lu + (-dist_mem_use); /* distribution step */
				loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1]);
				if (options->RowPerm != NO)
					loc_max = SUPERLU_MAX(loc_max, GA_mem_use);
			}
			else
			{																				/* Serial symbolic. GA_mem_use is for global A */
				mem_stage[0] = symb_mem_usage.total + GA_mem_use;							/* symbfact step */
				mem_stage[1] = symb_mem_usage.for_lu + dist_mem_use + num_mem_usage.for_lu; /* distribution step */
				loc_max = SUPERLU_MAX(mem_stage[0], mem_stage[1]);
			}

			mem_stage[2] = num_mem_usage.total;			  /* numerical factorization step */
			loc_max = SUPERLU_MAX(loc_max, mem_stage[2]); /* local max of 3 stages */

			local_struct.val = loc_max;
			local_struct.rank = grid->iam;
			MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm);
			int all_highmark_rank = global_struct.rank;
			float all_highmark_mem = global_struct.val * 1e-6;

			MPI_Reduce(&loc_max, &avg,
					   1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce(&num_mem_usage.for_lu, &for_lu,
					   1, MPI_FLOAT, MPI_SUM, 0, grid->comm);
			MPI_Reduce(&num_mem_usage.total, &total,
					   1, MPI_FLOAT, MPI_SUM, 0, grid->comm);

			/*-- Compute memory usage of numerical factorization --*/
			local_struct.val = num_mem_usage.for_lu;
			MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm);
			int lu_max_rank = global_struct.rank;
			float lu_max_mem = global_struct.val * 1e-6;

			local_struct.val = stat->peak_buffer;
			MPI_Reduce(&local_struct, &global_struct, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, grid->comm);
			int buffer_peak_rank = global_struct.rank;
			float buffer_peak = global_struct.val * 1e-6;

			if (iam == 0)
			{
				printf("** Total highmark (MB):\n"
					   "    Sum-of-all : %8.2f | Avg : %8.2f  | Max : %8.2f\n",
					   avg * 1e-6,
					   avg / grid->nprow / grid->npcol * 1e-6,
					   all_highmark_mem);
				printf("    Max at rank %d, different stages (MB):\n"
					   "\t. symbfact        %8.2f\n"
					   "\t. distribution    %8.2f\n"
					   "\t. numfact         %8.2f\n",
					   all_highmark_rank, mem_stage[0] * 1e-6, mem_stage[1] * 1e-6, mem_stage[2] * 1e-6);

				printf("** NUMfact space (MB): (sum-of-all-processes)\n"
					   "    L\\U :        %8.2f |  Total : %8.2f\n",
					   for_lu * 1e-6, total * 1e-6);
				printf("\t. max at rank %d, max L+U memory (MB): %8.2f\n"
					   "\t. max at rank %d, peak buffer (MB):    %8.2f\n",
					   lu_max_rank, lu_max_mem,
					   buffer_peak_rank, buffer_peak);
				printf("**************************************************\n\n");
				printf("** number of Tiny Pivots: %8d\n\n", stat->TinyPivots);
				fflush(stdout);
			}
		} /* end printing stats */

/* nvshmem related. The nvshmem_malloc has to be called before dtrs_compute_communication_structure, otherwise solve is much slower*/
#ifdef HAVE_NVSHMEM
		nsupers = Glu_persist->supno[n - 1] + 1;
		int nc = CEILING(nsupers, grid->npcol);
		int nr = CEILING(nsupers, grid->nprow);
		int flag_bc_size = RDMA_FLAG_SIZE * (nc + 1);
		int flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
		int my_flag_bc_size = RDMA_FLAG_SIZE * (nc + 1);
		int my_flag_rd_size = RDMA_FLAG_SIZE * nr * 2;
		int maxrecvsz = sp_ienv_dist(3, options) * nrhs + SUPERLU_MAX(XK_H, LSUM_H);
		int ready_x_size = maxrecvsz * nc;
		int ready_lsum_size = 2 * maxrecvsz * nr;
		if (get_acc_solve())
		{
			nv_init_wrapper(grid->comm);
			dprepare_multiGPU_buffers(flag_bc_size, flag_rd_size, ready_x_size, ready_lsum_size, my_flag_bc_size, my_flag_rd_size);
		}
#endif

		if (options->Fact != SamePattern_SameRowPerm)
		{
			nsupers = Glu_persist->supno[n - 1] + 1;
			int *supernodeMask = int32Malloc_dist(nsupers);
			for (int ii = 0; ii < nsupers; ii++)
				supernodeMask[ii] = 1;
			dtrs_compute_communication_structure(options, n, LUstruct,
												 ScalePermstruct, supernodeMask, grid, stat);
			SUPERLU_FREE(supernodeMask);
		}

	} /* end if (!factored) */

	if (options->Fact == DOFACT || options->Fact == SamePattern)
	{
		/* Need to reset the solve's communication pattern,
		   because perm_r[] and/or perm_c[] is changed.    */
		if (options->SolveInitialized == YES)
		{										  /* Initialized before */
			dSolveFinalize(options, SOLVEstruct); /* Clean up structure */
			pdgstrs_delete_device_lsum_x(SOLVEstruct);
			options->SolveInitialized = NO; /* Reset the solve state */
		}
	}

#if (defined(GPU_ACC))
	if (options->DiagInv == NO && get_acc_solve())
	{
		if (iam == 0)
		{
			printf("!!WARNING: GPU trisolve requires setting options->DiagInv==YES\n");
			printf("           otherwise, use CPU trisolve\n");
			fflush(stdout);
		}
		// exit(0);  // Sherry: need to return an error flag
	}
#endif

	if (options->DiagInv == YES && (Fact != FACTORED))
	{
		pdCompute_Diag_Inv(n, LUstruct, grid, stat, info);
		int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
#ifdef GPU_ACC
		pdconvertU(options, grid, LUstruct, stat, n);
#endif

#ifdef GPU_ACC
		checkGPU(gpuMemcpy(LUstruct->Llu->d_Linv_bc_dat, LUstruct->Llu->Linv_bc_dat,
						   (LUstruct->Llu->Linv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
		checkGPU(gpuMemcpy(LUstruct->Llu->d_Uinv_bc_dat, LUstruct->Llu->Uinv_bc_dat,
						   (LUstruct->Llu->Uinv_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
		checkGPU(gpuMemcpy(LUstruct->Llu->d_Lnzval_bc_dat, LUstruct->Llu->Lnzval_bc_dat,
						   (LUstruct->Llu->Lnzval_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));
#endif
	}

	/* ------------------------------------------------------------
	   Compute the solution matrix X.
	   ------------------------------------------------------------*/
	if (nrhs && *info == 0)
	{

		if (!(b_work = doubleMalloc_dist(n)))
			ABORT("Malloc fails for b_work[]");

		/* ------------------------------------------------------------
		   Scale the right-hand side if equilibration was performed.
		   ------------------------------------------------------------*/
		if (notran)
		{
			if (rowequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						b_col[i] *= R[irow];
						++irow;
					}
					b_col += ldb;
				}
			}
		}
		else if (colequ)
		{
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				irow = fst_row;
				for (i = 0; i < m_loc; ++i)
				{
					b_col[i] *= C[irow];
					++irow;
				}
				b_col += ldb;
			}
		}

		/* Save a copy of the right-hand side. */
		ldx = ldb;
		if (!(X = doubleMalloc_dist(((size_t)ldx) * nrhs)))
			ABORT("Malloc fails for X[]");
		x_col = X;
		b_col = B;
		for (j = 0; j < nrhs; ++j)
		{
			memcpy(x_col, b_col, m_loc * sizeof(double));
			x_col += ldx;
			b_col += ldb;
		}

		/* ------------------------------------------------------------
		   Solve the linear system.
		   ------------------------------------------------------------*/
		if (options->SolveInitialized == NO)
		{ /* First time */
			dSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct, grid,
					   SOLVEstruct);
			/* Inside this routine, SolveInitialized is set to YES.
		   For repeated call to pdgssvx(), no need to re-initialilze
		   the Solve data & communication structures, unless a new
		   factorization with Fact == DOFACT or SamePattern is asked for. */

			if (get_acc_solve())
			{
				int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
				int *supernodeMask = int32Malloc_dist(nsupers);
				for (int ii = 0; ii < nsupers; ii++)
					supernodeMask[ii] = 1;
				pdgstrs_init_device_lsum_x(options, n, m_loc, nrhs, grid, LUstruct, SOLVEstruct, supernodeMask);
				SUPERLU_FREE(supernodeMask);
			}
		}

		pdgstrs(options, n, LUstruct, ScalePermstruct, grid, X, m_loc,
				fst_row, ldb, nrhs, SOLVEstruct, stat, info);

		/* ------------------------------------------------------------
		   Use iterative refinement to improve the computed solution and
		   compute error bounds and backward error estimates for it.
		   ------------------------------------------------------------*/
		if (options->IterRefine)
		{
			/* Improve the solution by iterative refinement. */
			int_t *it;
			int_t *colind_gsmv = SOLVEstruct->A_colind_gsmv;
			/* This was allocated and set to NULL in dSolveInit() */
			dSOLVEstruct_t *SOLVEstruct1; /* Used by refinement. */

			t = SuperLU_timer_();
			if (options->RefineInitialized == NO || Fact == DOFACT)
			{
				/* All these cases need to re-initialize gsmv structure */
				if (options->RefineInitialized)
					pdgsmv_finalize(SOLVEstruct->gsmv_comm);
				pdgsmv_init(A, SOLVEstruct->row_to_proc, grid,
							SOLVEstruct->gsmv_comm);

				/* Save a copy of the transformed local col indices
		   in colind_gsmv[]. */
				if (colind_gsmv)
					SUPERLU_FREE(colind_gsmv);
				if (!(it = intMalloc_dist(nnz_loc)))
					ABORT("Malloc fails for colind_gsmv[]");
				colind_gsmv = SOLVEstruct->A_colind_gsmv = it;
				for (i = 0; i < nnz_loc; ++i)
					colind_gsmv[i] = colind[i];
				options->RefineInitialized = YES;
			}
			else if (Fact == SamePattern ||
					 Fact == SamePattern_SameRowPerm)
			{
				double atemp;
				int_t k, jcol, p;
				/* Swap to beginning the part of A corresponding to the
			   local part of X, as was done in pdgsmv_init() */
				for (i = 0; i < m_loc; ++i)
				{ /* Loop through each row */
					k = rowptr[i];
					for (j = rowptr[i]; j < rowptr[i + 1]; ++j)
					{
						jcol = colind[j];
						p = SOLVEstruct->row_to_proc[jcol];
						if (p == iam)
						{ /* Local */
							atemp = a[k];
							a[k] = a[j];
							a[j] = atemp;
							++k;
						}
					}
				}

				/* Re-use the local col indices of A obtained from the
			   previous call to pdgsmv_init() */
				for (i = 0; i < nnz_loc; ++i)
					colind[i] = colind_gsmv[i];
			}

			if (nrhs == 1)
			{ /* Use the existing solve structure */
				SOLVEstruct1 = SOLVEstruct;
			}
			else
			{ /* For nrhs > 1, since refinement is performed for RHS
		 one at a time, the communication structure for pdgstrs
		 is different than the solve with nrhs RHS.
		 So we use SOLVEstruct1 for the refinement step.
		  */
				if (!(SOLVEstruct1 = (dSOLVEstruct_t *)
						  SUPERLU_MALLOC(sizeof(dSOLVEstruct_t))))
					ABORT("Malloc fails for SOLVEstruct1");
				/* Copy the same stuff */
				SOLVEstruct1->row_to_proc = SOLVEstruct->row_to_proc;
				SOLVEstruct1->inv_perm_c = SOLVEstruct->inv_perm_c;
				SOLVEstruct1->num_diag_procs = SOLVEstruct->num_diag_procs;
				SOLVEstruct1->diag_procs = SOLVEstruct->diag_procs;
				SOLVEstruct1->diag_len = SOLVEstruct->diag_len;
				SOLVEstruct1->gsmv_comm = SOLVEstruct->gsmv_comm;
				SOLVEstruct1->A_colind_gsmv = SOLVEstruct->A_colind_gsmv;

				/* Initialize the *gstrs_comm for 1 RHS. */
				if (!(SOLVEstruct1->gstrs_comm = (pxgstrs_comm_t *)
						  SUPERLU_MALLOC(sizeof(pxgstrs_comm_t))))
					ABORT("Malloc fails for gstrs_comm[]");
				pdgstrs_init(n, m_loc, 1, fst_row, perm_r, perm_c, grid,
							 Glu_persist, SOLVEstruct1);
				if (get_acc_solve())
				{
					int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
					int *supernodeMask = int32Malloc_dist(nsupers);
					for (int ii = 0; ii < nsupers; ii++)
						supernodeMask[ii] = 1;
					pdgstrs_init_device_lsum_x(options, n, m_loc, 1, grid, LUstruct, SOLVEstruct1, supernodeMask);
					SUPERLU_FREE(supernodeMask);
				}
			}

			pdgsrfs(options, n, A, anorm, LUstruct, ScalePermstruct, grid,
					B, ldb, X, ldx, nrhs, SOLVEstruct1, berr, stat, info);

			/* Deallocate the storage associated with SOLVEstruct1 */
			if (nrhs > 1)
			{
				pdgstrs_delete_device_lsum_x(SOLVEstruct1);
				pxgstrs_finalize(SOLVEstruct1->gstrs_comm);
				SUPERLU_FREE(SOLVEstruct1);
			}

			stat->utime[REFINE] = SuperLU_timer_() - t;
		} /* end if IterRefine */

		/* Permute the solution matrix B <= Pc'*X. */
		pdPermute_Dense_Matrix(fst_row, m_loc, SOLVEstruct->row_to_proc,
							   SOLVEstruct->inv_perm_c,
							   X, ldx, B, ldb, nrhs, grid);
		/* Transform the solution matrix X to a solution of the original
		   system before equilibration. */
		if (notran)
		{
			if (colequ)
			{
				b_col = B;
				for (j = 0; j < nrhs; ++j)
				{
					irow = fst_row;
					for (i = 0; i < m_loc; ++i)
					{
						b_col[i] *= C[irow];
						++irow;
					}
					b_col += ldb;
				}
			}
		}
		else if (rowequ)
		{
			b_col = B;
			for (j = 0; j < nrhs; ++j)
			{
				irow = fst_row;
				for (i = 0; i < m_loc; ++i)
				{
					b_col[i] *= R[irow];
					++irow;
				}
				b_col += ldb;
			}
		}

		SUPERLU_FREE(b_work);
		SUPERLU_FREE(X);

	} /* end if nrhs != 0 && *info == 0 */

	/* Deallocate R and/or C if it was not used. */
	if (Equil && Fact != SamePattern_SameRowPerm)
	{
		switch (ScalePermstruct->DiagScale)
		{
		case NOEQUIL:
			SUPERLU_FREE(R);
			SUPERLU_FREE(C);
			break;
		case ROW:
			SUPERLU_FREE(C);
			break;
		case COL:
			SUPERLU_FREE(R);
			break;
		default:
			break;
		}
	}
} /* pdgssvx */

#ifdef GPU_ACC
void pdconvertU(superlu_dist_options_t *options, gridinfo_t *grid,
				dLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{
	int64_t nnz_ind, nnz_offset;
	int64_t nnz_val;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	int_t nsupers, nsupers_j, nsupers_i, ncol, ncol_loc, nrow;
	int_t lk, ik, ub, nub, i, il, gik, k, uptr, jj, ii, fnz, irow, jb;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t *Urbs = Llu->Urbs;
	int_t **Ucb_valptr = Llu->Ucb_valptr;		 /* Vertical linked list pointing to Unzval[] */
	Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr; /* Vertical linked list pointing to Uindex[] */
	int_t knsupc, iknsupc, ikfrow, iklrow;
	int_t *xsup = Glu_persist->xsup;
	;

	int iam = grid->iam;
	int mycol = MYCOL(iam, grid);
	int myrow = MYROW(iam, grid);

	int_t *usub, *usub1;
	double *uval;

	int64_t Ucolind_bc_cnt = 0, Uind_br_cnt = 0;
	int64_t Unzval_bc_cnt = 0, Unzval_br_cnt = 0;
	int64_t Uindval_loc_bc_cnt = 0;

	int_t next_lind; /* next available position in index[*] */
	int_t next_lval; /* next available position in nzval[*] */

	nsupers = Glu_persist->supno[n - 1] + 1;
	nsupers_j = CEILING(nsupers, grid->npcol); /* Number of local block columns */
	nsupers_i = CEILING(nsupers, grid->nprow); /* Number of local block rows */

	if (!(Llu->Ucolind_bc_ptr = (int_t **)SUPERLU_MALLOC(nsupers_j * sizeof(int_t *))))
		ABORT("Malloc fails for Llu->Ucolind_bc_ptr[].");
	Llu->Ucolind_bc_ptr[nsupers_j - 1] = NULL;

	if (!(Llu->Unzval_bc_ptr =
			  (double **)SUPERLU_MALLOC(nsupers_j * sizeof(double *))))
		ABORT("Malloc fails for Llu->Unzval_bc_ptr[].");
	Llu->Unzval_bc_ptr[nsupers_j - 1] = NULL;

	if (!(Llu->Uindval_loc_bc_ptr =
			  (int_t **)SUPERLU_MALLOC(nsupers_j * sizeof(int_t *))))
		ABORT("Malloc fails for Llu->Uindval_loc_bc_ptr[].");
	Llu->Uindval_loc_bc_ptr[nsupers_j - 1] = NULL;

	if (!(Llu->Uindval_loc_bc_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_j * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Uindval_loc_bc_offset[].");
	}
	Llu->Uindval_loc_bc_offset[nsupers_j - 1] = -1;

	if (!(Llu->Ucolind_bc_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_j * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_bc_offset[].");
	}
	Llu->Ucolind_bc_offset[nsupers_j - 1] = -1;

	if (!(Llu->Unzval_bc_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_j * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Lnzval_bc_offset[].");
	}
	Llu->Unzval_bc_offset[nsupers_j - 1] = -1;

	for (lk = 0; lk < nsupers_j; ++lk)
	{
		k = lk * grid->npcol + mycol; /* Global block number, col-wise. */
		knsupc = SuperSize(k);
		nub = Urbs[lk]; /* Number of U blocks in block column lk */

		if (nub > 0)
		{
			// First pass count sizes of Llu->Ucolind_bc_ptr[lk] and Llu->Unzval_bc_ptr[lk]
			nnz_ind = 0;
			nnz_val = 0;
			nnz_ind += BC_HEADER_NEWU;
			nrow = 0;
			for (ub = 0; ub < nub; ++ub)
			{
				ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
				usub = Llu->Ufstnz_br_ptr[ik];
				uval = Llu->Unzval_br_ptr[ik];
				i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
				i += UB_DESCRIPTOR;
				gik = ik * grid->nprow + myrow; /* Global block number, row-wise. */
				iknsupc = SuperSize(gik);
				nrow += iknsupc;
				ikfrow = FstBlockC(gik);
				iklrow = FstBlockC(gik + 1);
				uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */

				nnz_ind += UB_DESCRIPTOR_NEWU;

				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub[i + jj];
					if (fnz < iklrow)
					{ /* Nonzero segment. */
						nnz_val += iknsupc;
						nnz_ind += 1;
						Unzval_br_cnt += iklrow - fnz;
						// for (irow = fnz; irow < iklrow; ++irow)
						// dest[irow - ikfrow] -= uval[uptr++] * y[jj];
						// stat->ops[SOLVE] += 2 * (iklrow - fnz);
					}
				} /* for jj ... */
			} /* for ub ... */

			// Second pass fills Llu->Ucolind_bc_ptr[lk] and Llu->Unzval_bc_ptr[lk]
			if (!(Llu->Ucolind_bc_ptr[lk] = intMalloc_dist(nnz_ind + nrow * 2)))
				ABORT("Malloc fails for Llu->Ucolind_bc_ptr[lk]");
			Llu->Ucolind_bc_offset[lk] = nnz_ind + nrow * 2;
			Ucolind_bc_cnt += Llu->Ucolind_bc_offset[lk];

			if (!(Llu->Unzval_bc_ptr[lk] = doubleCalloc_dist(nnz_val)))
				ABORT("Calloc fails for Llu->Unzval_bc_ptr[lk].");
			Llu->Unzval_bc_offset[lk] = nnz_val;
			Unzval_bc_cnt += Llu->Unzval_bc_offset[lk];

			if (!(Llu->Uindval_loc_bc_ptr[lk] = intCalloc_dist(nub * 3)))
				ABORT("Malloc fails for Llu->Uindval_loc_bc_ptr[lk][]");
			Llu->Uindval_loc_bc_offset[lk] = nub * 3;
			Uindval_loc_bc_cnt += Llu->Uindval_loc_bc_offset[lk];

			Llu->Ucolind_bc_ptr[lk][0] = nub;
			Llu->Ucolind_bc_ptr[lk][1] = nrow;
			Llu->Ucolind_bc_ptr[lk][2] = nnz_ind;
			nnz_offset = nnz_ind;

			nnz_ind = 0;
			nnz_val = 0;
			ncol = 0;
			nnz_ind += BC_HEADER_NEWU;
			nrow = 0;
			for (ub = 0; ub < nub; ++ub)
			{
				ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
				usub = Llu->Ufstnz_br_ptr[ik];
				uval = Llu->Unzval_br_ptr[ik];
				i = Ucb_indptr[lk][ub].indpos; /* Start of the block in usub[]. */
				i += UB_DESCRIPTOR;
				gik = ik * grid->nprow + myrow; /* Global block number, row-wise. */
				iknsupc = SuperSize(gik);
				ikfrow = FstBlockC(gik);
				iklrow = FstBlockC(gik + 1);
				uptr = Ucb_valptr[lk][ub]; /* Start of the block in uval[]. */

				for (ii = 0; ii < iknsupc; ++ii)
				{
					Llu->Ucolind_bc_ptr[lk][nnz_offset + nrow * 2] = ub;
					Llu->Ucolind_bc_ptr[lk][nnz_offset + nrow * 2 + 1] = ii;
					nrow++;
				}

				ncol_loc = 0;
				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub[i + jj];
					if (fnz < iklrow)
					{																						  /* Nonzero segment. */
						Llu->Ucolind_bc_ptr[lk][nnz_ind + ncol_loc + UB_DESCRIPTOR_NEWU] = FstBlockC(k) + jj; /* Global column number */
						ncol_loc++;
						for (irow = fnz; irow < iklrow; ++irow)
						{
							Llu->Unzval_bc_ptr[lk][nnz_val + irow - ikfrow] = uval[uptr++];
							// if(lk==2){
							// printf("uval %5d %5d %5d %f %5d %5d \n",gik, uptr-1, irow - ikfrow, uval[uptr-1], Ucb_valptr[lk][ub],ub);
							// // printf("Unzval_bc_ptr %5d %f\n",gik, Llu->Unzval_bc_ptr[lk][nnz_val+irow - ikfrow]);
							// }
						}
						nnz_val += iknsupc;
					}
				} /* for jj ... */
				Llu->Ucolind_bc_ptr[lk][nnz_ind] = gik;
				Llu->Ucolind_bc_ptr[lk][nnz_ind + 1] = ncol_loc;

				Llu->Uindval_loc_bc_ptr[lk][ub] = ik;
				Llu->Uindval_loc_bc_ptr[lk][ub + nub] = nnz_ind;
				Llu->Uindval_loc_bc_ptr[lk][ub + nub * 2] = ncol;
				// if(lk==69)
				// 	printf("ub ncol_loc %5d %5d \n",ub, ncol_loc);
				ncol += ncol_loc * iknsupc;
				nnz_ind += ncol_loc + UB_DESCRIPTOR_NEWU;
			} /* for ub ... */
		}
		else
		{ /* nub <= 0 */
			Llu->Ucolind_bc_ptr[lk] = NULL;
			Llu->Unzval_bc_ptr[lk] = NULL;
			Llu->Ucolind_bc_offset[lk] = -1;
			Llu->Unzval_bc_offset[lk] = -1;
			Llu->Uindval_loc_bc_ptr[lk] = NULL;
			Llu->Uindval_loc_bc_offset[lk] = -1;
		}
	} /* end for lk ... */

	if (!(Llu->Uind_br_ptr = (int_t **)SUPERLU_MALLOC(nsupers_i * sizeof(int_t *))))
		ABORT("Malloc fails for Llu->Uind_br_ptr[].");
	Llu->Uind_br_ptr[nsupers_i - 1] = NULL;
	for (int_t ik = 0; ik < nsupers_i; ik++)
		if (!(Llu->Uind_br_ptr[ik] = intCalloc_dist(1)))
			ABORT("Malloc fails for Llu->Uind_br_ptr[ik]");

	if (!(Llu->Uind_br_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_i * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Uind_br_offset[].");
	}
	Llu->Uind_br_offset[nsupers_i - 1] = -1;

	// first pass count the number of nonzero supernodes per block row
	for (jb = 0; jb < nsupers_j; ++jb)
	{ /* for each block column ... */
		if (Llu->Ucolind_bc_ptr[jb] != NULL)
		{
			int_t nub = Llu->Ucolind_bc_ptr[jb][0];
			for (ub = 0; ub < nub; ub++)
			{
				ik = Llu->Uindval_loc_bc_ptr[jb][ub];
				Llu->Uind_br_ptr[ik][0]++;
			}
		}
	}
	for (int_t ik = 0; ik < nsupers_i; ik++)
	{
		int_t nub_r = Llu->Uind_br_ptr[ik][0];
		if (nub_r > 0)
		{
			SUPERLU_FREE(Llu->Uind_br_ptr[ik]);
			if (!(Llu->Uind_br_ptr[ik] = intCalloc_dist(nub_r * 2 + 1)))
				ABORT("Malloc fails for Llu->Uind_br_ptr[ik]");
			Llu->Uind_br_offset[ik] = nub_r * 2 + 1;
			Uind_br_cnt += Llu->Uind_br_offset[ik];
		}
		else
		{
			SUPERLU_FREE(Llu->Uind_br_ptr[ik]);
			Llu->Uind_br_ptr[ik] = NULL;
			Llu->Uind_br_offset[ik] = -1;
		}
	}
	// second pass fills Llu->Uind_br_ptr[ik]
	for (jb = 0; jb < nsupers_j; ++jb)
	{ /* for each block column ... */
		if (Llu->Ucolind_bc_ptr[jb] != NULL)
		{
			int_t nub = Llu->Ucolind_bc_ptr[jb][0];
			for (ub = 0; ub < nub; ub++)
			{
				ik = Llu->Uindval_loc_bc_ptr[jb][ub];
				Llu->Uind_br_ptr[ik][0]++;									// number of supernodes in this block row
				Llu->Uind_br_ptr[ik][Llu->Uind_br_ptr[ik][0] * 2 - 1] = jb; // the local column ID for each supernode
				Llu->Uind_br_ptr[ik][Llu->Uind_br_ptr[ik][0] * 2] = ub;		// the index in Ucolind_bc_ptr for each supernode
			}
		}
	}

/////////////////////////  row-wise data structure for GPU U solve (using the highest skyline accross the entire supernode row)
// Ucolind_br_ptr[lk][0]: number of nonempty supernode columns
// Ucolind_br_ptr[lk][1]: number of nonempty columns
// Ucolind_br_ptr[lk][2]: highest highest skyline accross the supernode row
// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP:UB_DESCRIPTOR_NEWUCPP+nub-1]: global supernodal ID
// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP+nub:UB_DESCRIPTOR_NEWUCPP+2*nub]: the starting index of each supernodal column in all nonempty columns
// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP+2*nub+1:UB_DESCRIPTOR_NEWUCPP+2*nub+ncol]: the global column id of each nonempty column
#ifdef U_BLOCK_PER_ROW_ROWDATA
	if (!(Llu->Ucolind_br_ptr = (int_t **)SUPERLU_MALLOC(nsupers_i * sizeof(int_t *))))
		ABORT("Malloc fails for Llu->Ucolind_br_ptr[].");
	Llu->Ucolind_br_ptr[nsupers_i - 1] = NULL;
	if (!(Llu->Ucolind_br_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_i * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_br_offset[].");
	}
	Llu->Ucolind_br_offset[nsupers_i - 1] = -1;

	if (!(Llu->Unzval_br_new_ptr =
			  (double **)SUPERLU_MALLOC(nsupers_i * sizeof(double *))))
		ABORT("Malloc fails for Llu->Unzval_br_new_ptr[].");
	Llu->Unzval_br_new_ptr[nsupers_i - 1] = NULL;
	if (!(Llu->Unzval_br_new_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_i * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Unzval_br_new_offset[].");
	}
	Llu->Unzval_br_new_offset[nsupers_i - 1] = -1;

	int64_t Ucolind_br_cnt = 0;
	int64_t Unzval_br_new_cnt = 0;
	for (lk = 0; lk < nsupers_i; ++lk)
	{									/* For each block row. */
		gik = lk * grid->nprow + myrow; /* Global block number, row-wise. */
		iknsupc = SuperSize(gik);
		usub1 = Llu->Ufstnz_br_ptr[lk];
		uval = Llu->Unzval_br_ptr[lk];
		if (usub1)
		{				   /* Not an empty block row. */
			i = BR_HEADER; /* Pointer in index array. */
			uptr = 0;	   /* Pointer in nzval array. */
			int_t nub = usub1[0];
			int_t ncol = 0;
			int_t LDA = 0;
			ikfrow = FstBlockC(gik);
			iklrow = FstBlockC(gik + 1);
			for (int_t lb = 0; lb < nub; ++lb)
			{				  /* For all column blocks. */
				k = usub1[i]; /* Global block number, column-wise. */
				knsupc = SuperSize(k);
				i += UB_DESCRIPTOR;
				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub1[i + jj];
					if (fnz < iklrow)
					{
						LDA = SUPERLU_MAX(LDA, iklrow - fnz);
						ncol += 1;
					}
				} /* for jj ... */
				i += knsupc;
			}
			if (!(Llu->Ucolind_br_ptr[lk] = intMalloc_dist(UB_DESCRIPTOR_NEWUCPP + ncol + nub * 2 + 1)))
				ABORT("Malloc fails for Llu->Ucolind_br_ptr[lk]");
			Llu->Ucolind_br_offset[lk] = UB_DESCRIPTOR_NEWUCPP + ncol + nub * 2 + 1;
			Ucolind_br_cnt += Llu->Ucolind_br_offset[lk];

			if (!(Llu->Unzval_br_new_ptr[lk] = doubleCalloc_dist(ncol * LDA)))
				ABORT("Calloc fails for Llu->Unzval_br_new_ptr[lk].");
			Llu->Unzval_br_new_offset[lk] = ncol * LDA;
			Unzval_br_new_cnt += Llu->Unzval_br_new_offset[lk];

			Llu->Ucolind_br_ptr[lk][0] = nub;
			Llu->Ucolind_br_ptr[lk][1] = ncol;
			Llu->Ucolind_br_ptr[lk][2] = LDA;

			ncol = 0;
			i = BR_HEADER; /* Pointer in index array. */
			Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + nub] = 0;
			for (int_t lb = 0; lb < nub; ++lb)
			{				  /* For all column blocks. */
				k = usub1[i]; /* Global block number, column-wise. */

				knsupc = SuperSize(k);
				Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + lb] = k;

				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub1[i + jj + UB_DESCRIPTOR];
					if (fnz < iklrow)
					{
						Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + 2 * nub + 1 + ncol] = FstBlockC(k) + jj; /* Global column number */
						for (irow = fnz; irow < iklrow; ++irow)
						{
							Llu->Unzval_br_new_ptr[lk][ncol * LDA + irow - (ikfrow + iknsupc - LDA)] = uval[uptr++];
						}
						ncol += 1;
					}
				} /* for jj ... */
				// j += usub1[i+1];       /* number of nonzeros for this block in Llu->Unzval_br_ptr[lk]*/
				i += knsupc + UB_DESCRIPTOR;
				Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + nub + lb + 1] = ncol;
			}
		}
		else
		{
			Llu->Ucolind_br_ptr[lk] = NULL;
			Llu->Unzval_br_new_ptr[lk] = NULL;
			Llu->Ucolind_br_offset[lk] = -1;
			Llu->Unzval_br_new_offset[lk] = -1;
		}
	}
#endif

	// safe guard
#ifdef U_BLOCK_PER_ROW_ROWDATA
	Ucolind_br_cnt += 1;
	Unzval_br_new_cnt += 1;
#endif

	Uind_br_cnt += 1;
	Ucolind_bc_cnt += 1;
	Unzval_bc_cnt += 1;
	Uindval_loc_bc_cnt += 1;
	if (!(Llu->Ucolind_bc_dat =
			  (int_t *)SUPERLU_MALLOC(Ucolind_bc_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_bc_dat[].");
	}
	if (!(Llu->Unzval_bc_dat =
			  (double *)SUPERLU_MALLOC(Unzval_bc_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Llu->Unzval_bc_dat[].");
	}
	if (!(Llu->Uindval_loc_bc_dat =
			  (int_t *)SUPERLU_MALLOC(Uindval_loc_bc_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Uindval_loc_bc_dat[].");
	}
	if (!(Llu->Uind_br_dat =
			  (int_t *)SUPERLU_MALLOC(Uind_br_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Uind_br_dat[].");
	}

#ifdef U_BLOCK_PER_ROW_ROWDATA
	if (!(Llu->Ucolind_br_dat =
			  (int_t *)SUPERLU_MALLOC(Ucolind_br_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_br_dat[].");
	}
	if (!(Llu->Unzval_br_new_dat =
			  (double *)SUPERLU_MALLOC(Unzval_br_new_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Llu->Unzval_br_new_dat[].");
	}
#endif
	/* use contingous memory for Ucolind_bc_ptr, Unzval_bc_ptr, Uindval_loc_bc_ptr*/
	k = CEILING(nsupers, grid->npcol); /* Number of local block columns */

#ifdef U_BLOCK_PER_ROW_ROWDATA
	Ucolind_br_cnt = 0;
	Unzval_br_new_cnt = 0;
#endif
	Ucolind_bc_cnt = 0;
	Uind_br_cnt = 0;
	Unzval_bc_cnt = 0;
	Uindval_loc_bc_cnt = 0;

	int64_t tmp_cnt;

	for (jb = 0; jb < k; ++jb)
	{ /* for each block column ... */
		if (Llu->Ucolind_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Llu->Ucolind_bc_offset[jb]; ++jj)
			{
				Llu->Ucolind_bc_dat[Ucolind_bc_cnt + jj] = Llu->Ucolind_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Llu->Ucolind_bc_ptr[jb]);
			Llu->Ucolind_bc_ptr[jb] = &Llu->Ucolind_bc_dat[Ucolind_bc_cnt];
			tmp_cnt = Llu->Ucolind_bc_offset[jb];
			Llu->Ucolind_bc_offset[jb] = Ucolind_bc_cnt;
			Ucolind_bc_cnt += tmp_cnt;
		}

		if (Llu->Unzval_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Llu->Unzval_bc_offset[jb]; ++jj)
			{
				Llu->Unzval_bc_dat[Unzval_bc_cnt + jj] = Llu->Unzval_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Llu->Unzval_bc_ptr[jb]);
			Llu->Unzval_bc_ptr[jb] = &Llu->Unzval_bc_dat[Unzval_bc_cnt];
			tmp_cnt = Llu->Unzval_bc_offset[jb];
			Llu->Unzval_bc_offset[jb] = Unzval_bc_cnt;
			Unzval_bc_cnt += tmp_cnt;
		}

		if (Llu->Uindval_loc_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Llu->Uindval_loc_bc_offset[jb]; ++jj)
			{
				Llu->Uindval_loc_bc_dat[Uindval_loc_bc_cnt + jj] = Llu->Uindval_loc_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Llu->Uindval_loc_bc_ptr[jb]);
			Llu->Uindval_loc_bc_ptr[jb] = &Llu->Uindval_loc_bc_dat[Uindval_loc_bc_cnt];
			tmp_cnt = Llu->Uindval_loc_bc_offset[jb];
			Llu->Uindval_loc_bc_offset[jb] = Uindval_loc_bc_cnt;
			Uindval_loc_bc_cnt += tmp_cnt;
		}

	} /* end for jb ... */

	k = CEILING(nsupers, grid->nprow); /* Number of local block rows */
	for (int_t ib = 0; ib < k; ++ib)
	{ /* for each block row ... */
		if (Llu->Uind_br_ptr[ib] != NULL)
		{
			for (ii = 0; ii < Llu->Uind_br_offset[ib]; ++ii)
			{
				Llu->Uind_br_dat[Uind_br_cnt + ii] = Llu->Uind_br_ptr[ib][ii];
			}
			SUPERLU_FREE(Llu->Uind_br_ptr[ib]);
			Llu->Uind_br_ptr[ib] = &Llu->Uind_br_dat[Uind_br_cnt];
			tmp_cnt = Llu->Uind_br_offset[ib];
			Llu->Uind_br_offset[ib] = Uind_br_cnt;
			// printf("ib %5d Llu->Uind_br_offset[ib] %5d\n",ib,Llu->Uind_br_offset[ib]);
			Uind_br_cnt += tmp_cnt;
		}
#ifdef U_BLOCK_PER_ROW_ROWDATA
		if (Llu->Ucolind_br_ptr[ib] != NULL)
		{
			for (ii = 0; ii < Llu->Ucolind_br_offset[ib]; ++ii)
			{
				Llu->Ucolind_br_dat[Ucolind_br_cnt + ii] = Llu->Ucolind_br_ptr[ib][ii];
			}
			SUPERLU_FREE(Llu->Ucolind_br_ptr[ib]);
			Llu->Ucolind_br_ptr[ib] = &Llu->Ucolind_br_dat[Ucolind_br_cnt];
			tmp_cnt = Llu->Ucolind_br_offset[ib];
			Llu->Ucolind_br_offset[ib] = Ucolind_br_cnt;
			// printf("ib %5d Llu->Ucolind_br_offset[ib] %5d\n",ib,Llu->Ucolind_br_offset[ib]);
			Ucolind_br_cnt += tmp_cnt;
		}

		if (Llu->Unzval_br_new_ptr[ib] != NULL)
		{
			for (ii = 0; ii < Llu->Unzval_br_new_offset[ib]; ++ii)
			{
				Llu->Unzval_br_new_dat[Unzval_br_new_cnt + ii] = Llu->Unzval_br_new_ptr[ib][ii];
			}
			SUPERLU_FREE(Llu->Unzval_br_new_ptr[ib]);
			Llu->Unzval_br_new_ptr[ib] = &Llu->Unzval_br_new_dat[Unzval_br_new_cnt];
			tmp_cnt = Llu->Unzval_br_new_offset[ib];
			Llu->Unzval_br_new_offset[ib] = Unzval_br_new_cnt;
			// printf("ib %5d Llu->Unzval_br_new_offset[ib] %5d\n",ib,Llu->Unzval_br_new_offset[ib]);
			Unzval_br_new_cnt += tmp_cnt;
		}
#endif
	}

#ifdef U_BLOCK_PER_ROW_ROWDATA
	Llu->Unzval_br_new_cnt = Unzval_br_new_cnt;
	Llu->Ucolind_br_cnt = Ucolind_br_cnt;
#endif
	Llu->Uind_br_cnt = Uind_br_cnt;
	Llu->Ucolind_bc_cnt = Ucolind_bc_cnt;
	Llu->Unzval_bc_cnt = Unzval_bc_cnt;
	Llu->Uindval_loc_bc_cnt = Uindval_loc_bc_cnt;

	if (get_acc_solve())
	{
		// checkGPU(gpuFree(Llu->d_Ucolind_bc_dat));
		// checkGPU(gpuFree(Llu->d_Ucolind_bc_offset));
		// checkGPU(gpuFree(Llu->d_Unzval_bc_dat));
		// checkGPU(gpuFree(Llu->d_Unzval_bc_offset));
		// checkGPU(gpuFree(Llu->d_Uindval_loc_bc_dat));
		// checkGPU(gpuFree(Llu->d_Uindval_loc_bc_offset));

		if (Llu->d_Ucolind_bc_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Ucolind_bc_dat, (Llu->Ucolind_bc_cnt) * sizeof(int_t)));
		checkGPU(gpuMemcpy(Llu->d_Ucolind_bc_dat, Llu->Ucolind_bc_dat, (Llu->Ucolind_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
		if (Llu->d_Ucolind_bc_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Ucolind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Ucolind_bc_offset, Llu->Ucolind_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t), gpuMemcpyHostToDevice));

		if (Llu->d_Uind_br_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Uind_br_dat, (Llu->Uind_br_cnt) * sizeof(int_t)));
		checkGPU(gpuMemcpy(Llu->d_Uind_br_dat, Llu->Uind_br_dat, (Llu->Uind_br_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
		if (Llu->d_Uind_br_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Uind_br_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Uind_br_offset, Llu->Uind_br_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t), gpuMemcpyHostToDevice));

#ifdef U_BLOCK_PER_ROW_ROWDATA
		if (Llu->d_Ucolind_br_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Ucolind_br_dat, (Llu->Ucolind_br_cnt) * sizeof(int_t)));
		checkGPU(gpuMemcpy(Llu->d_Ucolind_br_dat, Llu->Ucolind_br_dat, (Llu->Ucolind_br_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
		if (Llu->d_Ucolind_br_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Ucolind_br_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Ucolind_br_offset, Llu->Ucolind_br_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t), gpuMemcpyHostToDevice));

		if (Llu->d_Unzval_br_new_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Unzval_br_new_dat, (Llu->Unzval_br_new_cnt) * sizeof(double)));
		checkGPU(gpuMemcpy(Llu->d_Unzval_br_new_dat, Llu->Unzval_br_new_dat, (Llu->Unzval_br_new_cnt) * sizeof(double), gpuMemcpyHostToDevice));
		if (Llu->d_Unzval_br_new_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Unzval_br_new_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Unzval_br_new_offset, Llu->Unzval_br_new_offset, CEILING(nsupers, grid->nprow) * sizeof(int64_t), gpuMemcpyHostToDevice));
#endif

		if (Llu->d_Unzval_bc_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Unzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Unzval_bc_offset, Llu->Unzval_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t), gpuMemcpyHostToDevice));
		if (Llu->d_Unzval_bc_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Unzval_bc_dat, (Llu->Unzval_bc_cnt) * sizeof(double)));
		checkGPU(gpuMemcpy(LUstruct->Llu->d_Unzval_bc_dat, LUstruct->Llu->Unzval_bc_dat, (LUstruct->Llu->Unzval_bc_cnt) * sizeof(double), gpuMemcpyHostToDevice));

		if (Llu->d_Uindval_loc_bc_dat == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Uindval_loc_bc_dat, (Llu->Uindval_loc_bc_cnt) * sizeof(int_t)));
		checkGPU(gpuMemcpy(Llu->d_Uindval_loc_bc_dat, Llu->Uindval_loc_bc_dat, (Llu->Uindval_loc_bc_cnt) * sizeof(int_t), gpuMemcpyHostToDevice));
		if (Llu->d_Uindval_loc_bc_offset == NULL)
			checkGPU(gpuMalloc((void **)&Llu->d_Uindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t)));
		checkGPU(gpuMemcpy(Llu->d_Uindval_loc_bc_offset, Llu->Uindval_loc_bc_offset, CEILING(nsupers, grid->npcol) * sizeof(int64_t), gpuMemcpyHostToDevice));
	}
#ifdef U_BLOCK_PER_ROW_ROWDATA
	SUPERLU_FREE(Llu->Ucolind_br_dat);
	SUPERLU_FREE(Llu->Ucolind_br_offset);
	SUPERLU_FREE(Llu->Unzval_br_new_dat);
	SUPERLU_FREE(Llu->Unzval_br_new_offset);
#endif

	SUPERLU_FREE(Llu->Uind_br_ptr);
	SUPERLU_FREE(Llu->Uind_br_dat);
	SUPERLU_FREE(Llu->Uind_br_offset);
	SUPERLU_FREE(Llu->Ucolind_bc_ptr);
	SUPERLU_FREE(Llu->Ucolind_bc_dat);
	SUPERLU_FREE(Llu->Ucolind_bc_offset);
	SUPERLU_FREE(Llu->Unzval_bc_dat);
	SUPERLU_FREE(Llu->Unzval_bc_ptr);
	SUPERLU_FREE(Llu->Unzval_bc_offset);
	SUPERLU_FREE(Llu->Uindval_loc_bc_ptr);
	SUPERLU_FREE(Llu->Uindval_loc_bc_dat);
	SUPERLU_FREE(Llu->Uindval_loc_bc_offset);

} /* pdconvertU */
#endif /* ifdef GPU_ACC */

void dpacked2skyline(int_t k, int_t *usubpack, double *valpack, int_t *usub, double *uval, int_t *xsup)
{
	int_t kLastRow = xsup[k + 1];
	int_t srcUvalPtr = 0;
	int_t dstUvalPtr = 0;
	// reset the USUB ptr
	int_t usubPtr = BR_HEADER;
	int_t nub = usubpack[0];
	int_t kSupSz = usubpack[2];

	for (int_t ub = 0; ub < nub; ub++)
	{
		int_t gblockId = usub[usubPtr];
		int_t gsupc = SuperSize(gblockId);
		for (int_t col = 0; col < gsupc; col++)
		{
			int_t segsize = kLastRow - usub[usubPtr + UB_DESCRIPTOR + col];
			if (segsize)
			{
				for (int row = 0; row < kSupSz; row++)
				{
					if (row < kSupSz - segsize)
						dstUvalPtr++;
					else
						uval[srcUvalPtr++] = valpack[dstUvalPtr++];
				}
			}
		}

		usubPtr += UB_DESCRIPTOR + gsupc;
	}
}

void pdconvertUROWDATA2skyline(superlu_dist_options_t *options, gridinfo_t *grid,
							   dLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{

	int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
	double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;
	int_t nsupers = getNsupers(n, LUstruct->Glu_persist);
	int iam = grid->iam;
	int mycol = MYCOL(iam, grid);
	int myrow = MYROW(iam, grid);
	int Pr = grid->nprow;
	int Pc = grid->npcol;
	int_t *xsup = LUstruct->Glu_persist->xsup;

	for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
	{
		if (Ufstnz_br_ptr[i] != NULL)
		{
			int_t globalId = i * Pr + myrow;
			dpacked2skyline(globalId, LUstruct->Llu->Ucolind_br_ptr[i], LUstruct->Llu->Unzval_br_new_ptr[i], Ufstnz_br_ptr[i], Unzval_br_ptr[i], xsup);
		}
	}
}

void pdconvert_flatten_skyline2UROWDATA(superlu_dist_options_t *options, gridinfo_t *grid,
										dLUstruct_t *LUstruct, SuperLUStat_t *stat, int n)
{
	int64_t nnz_ind, nnz_offset;
	int64_t nnz_val;
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	int_t nsupers, nsupers_j, nsupers_i, ncol, ncol_loc, nrow;
	int_t lk, ik, ub, nub, i, il, gik, k, uptr, jj, ii, fnz, irow, jb;
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t *Urbs = Llu->Urbs;
	int_t **Ucb_valptr = Llu->Ucb_valptr;		 /* Vertical linked list pointing to Unzval[] */
	Ucb_indptr_t **Ucb_indptr = Llu->Ucb_indptr; /* Vertical linked list pointing to Uindex[] */
	int_t knsupc, iknsupc, ikfrow, iklrow;
	int_t *xsup = Glu_persist->xsup;
	;

	int iam = grid->iam;
	int mycol = MYCOL(iam, grid);
	int myrow = MYROW(iam, grid);

	int_t *usub, *usub1;
	double *uval;

	int64_t Ucolind_bc_cnt = 0, Uind_br_cnt = 0;
	int64_t Unzval_bc_cnt = 0, Unzval_br_cnt = 0;
	int64_t Uindval_loc_bc_cnt = 0;

	int_t next_lind; /* next available position in index[*] */
	int_t next_lval; /* next available position in nzval[*] */

	nsupers = Glu_persist->supno[n - 1] + 1;
	nsupers_j = CEILING(nsupers, grid->npcol); /* Number of local block columns */
	nsupers_i = CEILING(nsupers, grid->nprow); /* Number of local block rows */

	/////////////////////////  row-wise data structure (using the highest skyline accross the entire supernode row)
	// Ucolind_br_ptr[lk][0]: number of nonempty supernode columns
	// Ucolind_br_ptr[lk][1]: number of nonempty columns
	// Ucolind_br_ptr[lk][2]: highest highest skyline accross the supernode row
	// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP:UB_DESCRIPTOR_NEWUCPP+nub-1]: global supernodal ID
	// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP+nub:UB_DESCRIPTOR_NEWUCPP+2*nub]: the starting index of each supernodal column in all nonempty columns
	// Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP+2*nub+1:UB_DESCRIPTOR_NEWUCPP+2*nub+ncol]: the global column id of each nonempty column

	if (!(Llu->Ucolind_br_ptr = (int_t **)SUPERLU_MALLOC(nsupers_i * sizeof(int_t *))))
		ABORT("Malloc fails for Llu->Ucolind_br_ptr[].");
	Llu->Ucolind_br_ptr[nsupers_i - 1] = NULL;
	if (!(Llu->Ucolind_br_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_i * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_br_offset[].");
	}
	Llu->Ucolind_br_offset[nsupers_i - 1] = -1;

	if (!(Llu->Unzval_br_new_ptr =
			  (double **)SUPERLU_MALLOC(nsupers_i * sizeof(double *))))
		ABORT("Malloc fails for Llu->Unzval_br_new_ptr[].");
	Llu->Unzval_br_new_ptr[nsupers_i - 1] = NULL;
	if (!(Llu->Unzval_br_new_offset =
			  (int64_t *)SUPERLU_MALLOC(nsupers_i * sizeof(int64_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Unzval_br_new_offset[].");
	}
	Llu->Unzval_br_new_offset[nsupers_i - 1] = -1;

	int64_t tmp_cnt;
	int64_t Ucolind_br_cnt = 0;
	int64_t Unzval_br_new_cnt = 0;
	for (lk = 0; lk < nsupers_i; ++lk)
	{									/* For each block row. */
		gik = lk * grid->nprow + myrow; /* Global block number, row-wise. */
		iknsupc = SuperSize(gik);
		usub1 = Llu->Ufstnz_br_ptr[lk];
		uval = Llu->Unzval_br_ptr[lk];
		if (usub1)
		{				   /* Not an empty block row. */
			i = BR_HEADER; /* Pointer in index array. */
			uptr = 0;	   /* Pointer in nzval array. */
			int_t nub = usub1[0];
			int_t ncol = 0;
			int_t LDA = iknsupc;
			ikfrow = FstBlockC(gik);
			iklrow = FstBlockC(gik + 1);
			for (int_t lb = 0; lb < nub; ++lb)
			{				  /* For all column blocks. */
				k = usub1[i]; /* Global block number, column-wise. */
				knsupc = SuperSize(k);
				i += UB_DESCRIPTOR;
				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub1[i + jj];
					if (fnz < iklrow)
					{
						LDA = SUPERLU_MAX(LDA, iklrow - fnz);
						ncol += 1;
					}
				} /* for jj ... */
				i += knsupc;
			}
			if (!(Llu->Ucolind_br_ptr[lk] = intMalloc_dist(UB_DESCRIPTOR_NEWUCPP + ncol + nub * 2 + 1)))
				ABORT("Malloc fails for Llu->Ucolind_br_ptr[lk]");
			Llu->Ucolind_br_offset[lk] = UB_DESCRIPTOR_NEWUCPP + ncol + nub * 2 + 1;
			Ucolind_br_cnt += Llu->Ucolind_br_offset[lk];

			if (!(Llu->Unzval_br_new_ptr[lk] = doubleCalloc_dist(ncol * LDA)))
				ABORT("Calloc fails for Llu->Unzval_br_new_ptr[lk].");
			Llu->Unzval_br_new_offset[lk] = ncol * LDA;
			Unzval_br_new_cnt += Llu->Unzval_br_new_offset[lk];

			Llu->Ucolind_br_ptr[lk][0] = nub;
			Llu->Ucolind_br_ptr[lk][1] = ncol;
			Llu->Ucolind_br_ptr[lk][2] = LDA;

			ncol = 0;
			i = BR_HEADER; /* Pointer in index array. */
			Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + nub] = 0;
			for (int_t lb = 0; lb < nub; ++lb)
			{				  /* For all column blocks. */
				k = usub1[i]; /* Global block number, column-wise. */

				knsupc = SuperSize(k);
				Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + lb] = k;

				for (jj = 0; jj < knsupc; ++jj)
				{
					fnz = usub1[i + jj + UB_DESCRIPTOR];
					if (fnz < iklrow)
					{
						Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + 2 * nub + 1 + ncol] = FstBlockC(k) + jj; /* Global column number */
						for (irow = fnz; irow < iklrow; ++irow)
						{
							Llu->Unzval_br_new_ptr[lk][ncol * LDA + irow - (ikfrow + iknsupc - LDA)] = uval[uptr++];
						}
						ncol += 1;
					}
				} /* for jj ... */
				// j += usub1[i+1];       /* number of nonzeros for this block in Llu->Unzval_br_ptr[lk]*/
				i += knsupc + UB_DESCRIPTOR;
				Llu->Ucolind_br_ptr[lk][UB_DESCRIPTOR_NEWUCPP + nub + lb + 1] = ncol;
			}
		}
		else
		{
			Llu->Ucolind_br_ptr[lk] = NULL;
			Llu->Unzval_br_new_ptr[lk] = NULL;
			Llu->Ucolind_br_offset[lk] = -1;
			Llu->Unzval_br_new_offset[lk] = -1;
		}
	}

	// safe guard
	Ucolind_br_cnt += 1;
	Unzval_br_new_cnt += 1;

	if (!(Llu->Ucolind_br_dat =
			  (int_t *)SUPERLU_MALLOC(Ucolind_br_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Llu->Ucolind_br_dat[].");
	}
	if (!(Llu->Unzval_br_new_dat =
			  (double *)SUPERLU_MALLOC(Unzval_br_new_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Llu->Unzval_br_new_dat[].");
	}

	/* use contingous memory for Ucolind_bc_ptr, Unzval_bc_ptr, Uindval_loc_bc_ptr*/
	k = CEILING(nsupers, grid->npcol); /* Number of local block columns */

	Ucolind_br_cnt = 0;
	Unzval_br_new_cnt = 0;

	k = CEILING(nsupers, grid->nprow); /* Number of local block rows */
	for (int_t ib = 0; ib < k; ++ib)
	{ /* for each block row ... */

		if (Llu->Ucolind_br_ptr[ib] != NULL)
		{
			for (ii = 0; ii < Llu->Ucolind_br_offset[ib]; ++ii)
			{
				Llu->Ucolind_br_dat[Ucolind_br_cnt + ii] = Llu->Ucolind_br_ptr[ib][ii];
			}
			SUPERLU_FREE(Llu->Ucolind_br_ptr[ib]);
			Llu->Ucolind_br_ptr[ib] = &Llu->Ucolind_br_dat[Ucolind_br_cnt];
			tmp_cnt = Llu->Ucolind_br_offset[ib];
			Llu->Ucolind_br_offset[ib] = Ucolind_br_cnt;
			// printf("ib %5d Llu->Ucolind_br_offset[ib] %5d\n",ib,Llu->Ucolind_br_offset[ib]);
			Ucolind_br_cnt += tmp_cnt;
		}

		if (Llu->Unzval_br_new_ptr[ib] != NULL)
		{
			for (ii = 0; ii < Llu->Unzval_br_new_offset[ib]; ++ii)
			{
				Llu->Unzval_br_new_dat[Unzval_br_new_cnt + ii] = Llu->Unzval_br_new_ptr[ib][ii];
			}
			SUPERLU_FREE(Llu->Unzval_br_new_ptr[ib]);
			Llu->Unzval_br_new_ptr[ib] = &Llu->Unzval_br_new_dat[Unzval_br_new_cnt];
			tmp_cnt = Llu->Unzval_br_new_offset[ib];
			Llu->Unzval_br_new_offset[ib] = Unzval_br_new_cnt;
			// printf("ib %5d Llu->Unzval_br_new_offset[ib] %5d\n",ib,Llu->Unzval_br_new_offset[ib]);
			Unzval_br_new_cnt += tmp_cnt;
		}
	}

	Llu->Unzval_br_new_cnt = Unzval_br_new_cnt;
	Llu->Ucolind_br_cnt = Ucolind_br_cnt;

} /* pdconvert_flatten_skyline2UROWDATA */

int pdflatten_LDATA(superlu_dist_options_t *options, int_t n, dLUstruct_t *LUstruct,
					gridinfo_t *grid, SuperLUStat_t *stat)
{
	Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
	int kc;
	int nsupers = Glu_persist->supno[n - 1] + 1;
	int_t **rowlists, **collists;
	int_t *lsub;
	int_t jb, jj;

	kc = CEILING(nsupers, grid->npcol); /* Number of local block columns */
	int_t iam = grid->iam;
	int_t myrow = MYROW(iam, grid);
	int_t mycol = MYCOL(iam, grid);

	dLocalLU_t *Llu = LUstruct->Llu;
	int_t *xsup = Glu_persist->xsup;

	int_t len, len1, len2, nrbl;

	double **Lnzval_bc_ptr = Llu->Lnzval_bc_ptr; /* size ceil(NSUPERS/Pc) */
	double *Lnzval_bc_dat;						 /* size sum of sizes of Lnzval_bc_ptr[lk])                 */
	long int *Lnzval_bc_offset;					 /* size ceil(NSUPERS/Pc)                 */

	int_t **Lrowind_bc_ptr = Llu->Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
	int_t *Lrowind_bc_dat;						  /* size sum of sizes of Lrowind_bc_ptr[lk])                 */
	long int *Lrowind_bc_offset;				  /* size ceil(NSUPERS/Pc)                 */

	int_t **Lindval_loc_bc_ptr = Llu->Lindval_loc_bc_ptr; /* size ceil(NSUPERS/Pc)                 */
	int_t *Lindval_loc_bc_dat;							  /* size sum of sizes of Lindval_loc_bc_ptr[lk])                 */
	long int *Lindval_loc_bc_offset;					  /* size ceil(NSUPERS/Pc)                 */

	double **Linv_bc_ptr = Llu->Linv_bc_ptr; /* size ceil(NSUPERS/Pc) */
	double *Linv_bc_dat;					 /* size sum of sizes of Linv_bc_ptr[lk])                 */
	long int *Linv_bc_offset;				 /* size ceil(NSUPERS/Pc)                 */
	double **Uinv_bc_ptr = Llu->Uinv_bc_ptr; /* size ceil(NSUPERS/Pc) */
	double *Uinv_bc_dat;					 /* size sum of sizes of Uinv_bc_ptr[lk])                 */
	long int *Uinv_bc_offset;				 /* size ceil(NSUPERS/Pc) */

	////////////////////////////////////////////////////
	// use contignous memory for the L meta data
	int_t k = kc; /* Number of local block columns */
	long int Lnzval_bc_cnt = 0;
	long int Lrowind_bc_cnt = 0;
	long int Lindval_loc_bc_cnt = 0;
	long int Linv_bc_cnt = 0;
	long int Uinv_bc_cnt = 0;

	if (!(Lnzval_bc_offset =
			  (long int *)SUPERLU_MALLOC(k * sizeof(long int))))
	{
		fprintf(stderr, "Malloc fails for Lnzval_bc_offset[].");
	}
	Lnzval_bc_offset[k - 1] = -1;

	if (!(Lrowind_bc_offset =
			  (long int *)SUPERLU_MALLOC(k * sizeof(long int))))
	{
		fprintf(stderr, "Malloc fails for Lrowind_bc_offset[].");
	}
	Lrowind_bc_offset[k - 1] = -1;
	if (!(Lindval_loc_bc_offset =
			  (long int *)SUPERLU_MALLOC(k * sizeof(long int))))
	{
		fprintf(stderr, "Malloc fails for Lindval_loc_bc_offset[].");
	}
	Lindval_loc_bc_offset[k - 1] = -1;
	if (!(Linv_bc_offset =
			  (long int *)SUPERLU_MALLOC(k * sizeof(long int))))
	{
		fprintf(stderr, "Malloc fails for Linv_bc_offset[].");
	}
	Linv_bc_offset[k - 1] = -1;
	if (!(Uinv_bc_offset =
			  (long int *)SUPERLU_MALLOC(k * sizeof(long int))))
	{
		fprintf(stderr, "Malloc fails for Uinv_bc_offset[].");
	}
	Uinv_bc_offset[k - 1] = -1;

	for (int_t lk = 0; lk < k; ++lk)
	{
		jb = mycol + lk * grid->npcol; /* not sure */
		lsub = Lrowind_bc_ptr[lk];

		Linv_bc_offset[lk] = -1;
		Uinv_bc_offset[lk] = -1;
		Lrowind_bc_offset[lk] = -1;
		Lindval_loc_bc_offset[lk] = -1;
		Lnzval_bc_offset[lk] = -1;

		if (lsub)
		{
			nrbl = lsub[0]; /*number of L blocks */
			len = lsub[1];	/* LDA of the nzval[] */
			len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
			int_t nsupc = SuperSize(jb);
			len2 = nsupc * len;
			Lnzval_bc_offset[lk] = len2;
			Lnzval_bc_cnt += Lnzval_bc_offset[lk];

			Lrowind_bc_offset[lk] = len1;
			Lrowind_bc_cnt += Lrowind_bc_offset[lk];

			Lindval_loc_bc_offset[lk] = nrbl * 3;
			Lindval_loc_bc_cnt += Lindval_loc_bc_offset[lk];

			int_t krow = PROW(jb, grid);
			if (myrow == krow)
			{ /* diagonal block */
				Linv_bc_offset[lk] = nsupc * nsupc;
				Linv_bc_cnt += Linv_bc_offset[lk];
				Uinv_bc_offset[lk] = nsupc * nsupc;
				Uinv_bc_cnt += Uinv_bc_offset[lk];
			}
			else
			{
				Linv_bc_offset[lk] = -1;
				Uinv_bc_offset[lk] = -1;
			}
		}
	}

	Linv_bc_cnt += 1; // safe guard
	Uinv_bc_cnt += 1;
	Lrowind_bc_cnt += 1;
	Lindval_loc_bc_cnt += 1;
	Lnzval_bc_cnt += 1;
	if (!(Linv_bc_dat =
			  (double *)SUPERLU_MALLOC(Linv_bc_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Linv_bc_dat[].");
	}
	if (!(Uinv_bc_dat =
			  (double *)SUPERLU_MALLOC(Uinv_bc_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Uinv_bc_dat[].");
	}

	if (!(Lrowind_bc_dat =
			  (int_t *)SUPERLU_MALLOC(Lrowind_bc_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Lrowind_bc_dat[].");
	}
	if (!(Lindval_loc_bc_dat =
			  (int_t *)SUPERLU_MALLOC(Lindval_loc_bc_cnt * sizeof(int_t))))
	{
		fprintf(stderr, "Malloc fails for Lindval_loc_bc_dat[].");
	}
	if (!(Lnzval_bc_dat =
			  (double *)SUPERLU_MALLOC(Lnzval_bc_cnt * sizeof(double))))
	{
		fprintf(stderr, "Malloc fails for Lnzval_bc_dat[].");
	}

	/* use contingous memory for Linv_bc_ptr, Uinv_bc_ptr, Lrowind_bc_ptr, Lnzval_bc_ptr*/
	Linv_bc_cnt = 0;
	Uinv_bc_cnt = 0;
	Lrowind_bc_cnt = 0;
	Lnzval_bc_cnt = 0;
	Lindval_loc_bc_cnt = 0;
	long int tmp_cnt;
	for (jb = 0; jb < k; ++jb)
	{ /* for each block column ... */
		if (Linv_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Linv_bc_offset[jb]; ++jj)
			{
				Linv_bc_dat[Linv_bc_cnt + jj] = Linv_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Linv_bc_ptr[jb]);
			Linv_bc_ptr[jb] = &Linv_bc_dat[Linv_bc_cnt];
			tmp_cnt = Linv_bc_offset[jb];
			Linv_bc_offset[jb] = Linv_bc_cnt;
			Linv_bc_cnt += tmp_cnt;
		}

		if (Uinv_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Uinv_bc_offset[jb]; ++jj)
			{
				Uinv_bc_dat[Uinv_bc_cnt + jj] = Uinv_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Uinv_bc_ptr[jb]);
			Uinv_bc_ptr[jb] = &Uinv_bc_dat[Uinv_bc_cnt];
			tmp_cnt = Uinv_bc_offset[jb];
			Uinv_bc_offset[jb] = Uinv_bc_cnt;
			Uinv_bc_cnt += tmp_cnt;
		}

		if (Lrowind_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Lrowind_bc_offset[jb]; ++jj)
			{
				Lrowind_bc_dat[Lrowind_bc_cnt + jj] = Lrowind_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Lrowind_bc_ptr[jb]);
			Lrowind_bc_ptr[jb] = &Lrowind_bc_dat[Lrowind_bc_cnt];
			tmp_cnt = Lrowind_bc_offset[jb];
			Lrowind_bc_offset[jb] = Lrowind_bc_cnt;
			Lrowind_bc_cnt += tmp_cnt;
		}

		if (Lnzval_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Lnzval_bc_offset[jb]; ++jj)
			{
				Lnzval_bc_dat[Lnzval_bc_cnt + jj] = Lnzval_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Lnzval_bc_ptr[jb]);
			Lnzval_bc_ptr[jb] = &Lnzval_bc_dat[Lnzval_bc_cnt];
			tmp_cnt = Lnzval_bc_offset[jb];
			Lnzval_bc_offset[jb] = Lnzval_bc_cnt;
			Lnzval_bc_cnt += tmp_cnt;
		}

		if (Lindval_loc_bc_ptr[jb] != NULL)
		{
			for (jj = 0; jj < Lindval_loc_bc_offset[jb]; ++jj)
			{
				Lindval_loc_bc_dat[Lindval_loc_bc_cnt + jj] = Lindval_loc_bc_ptr[jb][jj];
			}
			SUPERLU_FREE(Lindval_loc_bc_ptr[jb]);
			Lindval_loc_bc_ptr[jb] = &Lindval_loc_bc_dat[Lindval_loc_bc_cnt];
			tmp_cnt = Lindval_loc_bc_offset[jb];
			Lindval_loc_bc_offset[jb] = Lindval_loc_bc_cnt;
			Lindval_loc_bc_cnt += tmp_cnt;
		}
	}

	Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
	Llu->Lrowind_bc_dat = Lrowind_bc_dat;
	Llu->Lrowind_bc_offset = Lrowind_bc_offset;
	Llu->Lrowind_bc_cnt = Lrowind_bc_cnt;

	Llu->Lindval_loc_bc_ptr = Lindval_loc_bc_ptr;
	Llu->Lindval_loc_bc_dat = Lindval_loc_bc_dat;
	Llu->Lindval_loc_bc_offset = Lindval_loc_bc_offset;
	Llu->Lindval_loc_bc_cnt = Lindval_loc_bc_cnt;

	Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
	Llu->Lnzval_bc_dat = Lnzval_bc_dat;
	Llu->Lnzval_bc_offset = Lnzval_bc_offset;
	Llu->Lnzval_bc_cnt = Lnzval_bc_cnt;

	Llu->Linv_bc_ptr = Linv_bc_ptr;
	Llu->Linv_bc_dat = Linv_bc_dat;
	Llu->Linv_bc_offset = Linv_bc_offset;
	Llu->Linv_bc_cnt = Linv_bc_cnt;

	Llu->Uinv_bc_ptr = Uinv_bc_ptr;
	Llu->Uinv_bc_dat = Uinv_bc_dat;
	Llu->Uinv_bc_offset = Uinv_bc_offset;
	Llu->Uinv_bc_cnt = Uinv_bc_cnt;

	return 0;
} // end pdflatten_LDATA
