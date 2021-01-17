// this is the subarray module
module j_systolic_array #(
    parameter SUBARRAY_WIDTH  = 2                     , // width of the subarray
    parameter SUBARRAY_HEIGHT = 2                     , // height of the subarray
    parameter GROUP_SIZE      = 16                    ,
    parameter DATA_WIDTH      = 3                     ,
    parameter WGT_WIDTH       = 3                     ,
    parameter MAX_BUDGET      = 16                    ,
    parameter IDX_WIDTH       = clog2(GROUP_SIZE)     ,
    parameter WGT_IDX_LEN     = MAX_BUDGET * IDX_WIDTH
) (
    input                                             clk                ,
    input                                             reset              ,
    input  [                  16*SUBARRAY_HEIGHT-1:0] accumulation_in_pos,
    input  [                  16*SUBARRAY_HEIGHT-1:0] accumulation_in_neg,
    input  [                     SUBARRAY_HEIGHT-1:0] mac_en             ,
    input  [DATA_WIDTH*GROUP_SIZE*SUBARRAY_WIDTH-1:0] dataflow_in        ,
    input  [           GROUP_SIZE*SUBARRAY_WIDTH-1:0] dataflow_sign_in   ,
    input  [                      SUBARRAY_WIDTH-1:0] update_w           ,
    input  [                      SUBARRAY_WIDTH-1:0] update_idx         ,
    output [                  16*SUBARRAY_HEIGHT-1:0] result             ,
    output [                     SUBARRAY_HEIGHT-1:0] result_en
);


    function [31:0] clog2 (input [31:0] x);
        reg [31:0] x_tmp;
        begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
        end
    endfunction

// create the systolic array
    wire [DATA_WIDTH*GROUP_SIZE-1:0] arr_dataflow_in        [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [           GROUP_SIZE-1:0] arr_dataflow_sign_in   [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [DATA_WIDTH*GROUP_SIZE-1:0] arr_dataflow_out       [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [           GROUP_SIZE-1:0] arr_dataflow_sign_out  [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_update_w_i         [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_update_w_o         [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_update_idx_i       [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_update_idx_o       [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_mac_en_i           [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire                             arr_mac_en_o           [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                     15:0] arr_accumulation_in_pos[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                     15:0] arr_accumulation_in_neg[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                     15:0] arr_result_pos         [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [                     15:0] arr_result_neg         [SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
    wire [   16*SUBARRAY_HEIGHT-1:0] result_pos                                                      ;
    wire [   16*SUBARRAY_HEIGHT-1:0] result_neg                                                      ;
 
    assign result = result_pos;
    genvar i,j;
    generate
        for (j=0; j<SUBARRAY_HEIGHT; j=j+1) begin: g_H
            for (i=0; i<SUBARRAY_WIDTH; i=i+1) begin: g_W

                assign arr_update_w_i[i][j] = update_w[i];
                assign arr_update_idx_i[i][j] = update_idx[i];

                if(j==0) begin: g_j_eq_0
                    assign arr_dataflow_in[i][0] = dataflow_in      [i*DATA_WIDTH*GROUP_SIZE +: DATA_WIDTH*GROUP_SIZE];
                    assign arr_dataflow_sign_in[i][0] = dataflow_sign_in      [i*GROUP_SIZE +: GROUP_SIZE];
                end else begin: g_j_others
                    assign arr_dataflow_in[i][j]      = arr_dataflow_out       [i][j-1];
                    assign arr_dataflow_sign_in[i][j] = arr_dataflow_sign_out       [i][j-1];
                end

                if(i==0) begin: g_i_eq_0
                    assign arr_accumulation_in_pos[0][j] = accumulation_in_pos  [16*j +: 16];
                    assign arr_accumulation_in_neg[0][j] = accumulation_in_neg  [16*j +: 16];
                    assign arr_mac_en_i[0][j]            = mac_en           [j +: 1];
                end else begin: g_i_others
                    assign arr_accumulation_in_pos[i][j] = arr_result_pos             [i-1][j];
                    assign arr_accumulation_in_neg[i][j] = arr_result_neg             [i-1][j];
                    assign arr_mac_en_i[i][j]            = arr_mac_en_o           [i-1][j];
                end

                if(i==(SUBARRAY_WIDTH-1)) begin: g_i_eq_W
                    assign result_pos[16*j+:16] = arr_result_pos             [SUBARRAY_WIDTH-1][j];
                    assign result_neg[16*j+:16] = arr_result_neg             [SUBARRAY_WIDTH-1][j];
                    assign result_en[j]  = arr_mac_en_o           [SUBARRAY_WIDTH-1][j];
                end

                mmac_new #(.GROUP_SIZE(GROUP_SIZE), .DATA_WIDTH(DATA_WIDTH), .WGT_WIDTH(WGT_WIDTH), .MAX_BUDGET(MAX_BUDGET), .IDX_WIDTH(IDX_WIDTH), .WGT_IDX_LEN(WGT_IDX_LEN)) i_j_MX_cell (
                    .clk          (clk                               ),
                    .reset        (reset                             ),
                    .pos_acc      (arr_accumulation_in_pos   [i][j]  ),
                    .neg_acc      (arr_accumulation_in_neg   [i][j]  ),
                    .update_w_i   (arr_update_w_i        [i][j]      ),
                    .mac_en_i     (arr_mac_en_i          [i][j]      ),
                    .update_idx_i (arr_update_idx_i        [i][j]    ),
                    .data_in      (arr_dataflow_in       [i][j]      ),
                    .data_sign_in (arr_dataflow_sign_in       [i][j] ),
                    .mac_en_o     (arr_mac_en_o          [i][j]      ),
                    .update_w_o   (arr_update_w_o        [i][j]      ),
                    .update_idx_o (arr_update_idx_o        [i][j]    ),
                    .data_out     (arr_dataflow_out      [i][j]      ),
                    .data_sign_out(arr_dataflow_sign_out       [i][j]),
                    .out_pos      (arr_result_pos            [i][j]  ),
                    .out_neg      (arr_result_neg            [i][j]  )
                );
            end
        end
    endgenerate
endmodule
