// Auto-generated TypeScript definitions for any_compute_ffi

export interface AnyComputeModule {
  /** Create a new empty VecSource. */
  anc_source_new(): number;
  /** Add a column definition to a VecSource. */
  anc_source_add_column(handle: number, name: string, kind: number): void;
  /** Push a row of integer values. */
  anc_source_push_row_ints(handle: number, values: number, len: number): void;
  /** Free a VecSource previously created by anc_source_new. */
  anc_source_free(handle: number): void;
}

export declare function loadAnyCompute(wasmUrl?: string): Promise<AnyComputeModule>;
