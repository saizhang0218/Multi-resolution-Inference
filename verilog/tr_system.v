module tr_system #(
    parameter SUBARRAY_WIDTH  = 8                            ,
    parameter SUBARRAY_HEIGHT = 8                            ,
    parameter WGT_SRAM_DEPTH  = 256*256                      ,
    parameter WGT_SRAM_ADDR_W = clog2(WGT_SRAM_DEPTH)        ,
    parameter XIN_SRAM_DEPTH  = 256*256                      ,
    parameter XIN_SRAM_ADDR_W = clog2(XIN_SRAM_DEPTH)        ,
    parameter N_XIN_PER_MX    = 8                            ,
    parameter N_WGT_PER_MX    = 8                            ,
    parameter N_WGT_MX        = SUBARRAY_WIDTH / N_WGT_PER_MX,
    parameter N_XIN_MX        = SUBARRAY_WIDTH/N_XIN_PER_MX  ,
    // parameters of the systolic array
    parameter GROUP_SIZE      = 8                            ,
    parameter DATA_WIDTH      = 3                            ,
    parameter WGT_WIDTH       = 3                            ,
    parameter MAX_BUDGET      = 16
) (
    input                                                 clk              ,
    input                                                 reset_n          ,
    // APB interface
    input                                                 psel             ,
    input       [                                   15:0] paddr            ,
    input                                                 pwrite           ,
    input       [                                   31:0] pwdata           ,
    input                                                 penable          ,
    //input       [            8*8*N_XIN_MX-1:0] xin_serial_output  , //
    //input [N_WGT_PER_MX*(WGT_WIDTH+1)*N_WGT_MX*GROUP_SIZE-1:0] wgt_serial_output,
    //input [                                  N_WGT_PER_MX-1:0] wgt_serial_en    ,
    //input [(WGT_WIDTH+1)*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_wgt ,
    //input [DATA_WIDTH*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_xin ,
    //input [SUBARRAY_WIDTH*8-1:0] sdr_encoder_out  ,
    //input [SUBARRAY_WIDTH*8-1:0] sdr_encoder_sign_out  ,
    //input [WGT_WIDTH*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in     ,
    //input [           GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_sign_in,
    //input [                      SUBARRAY_WIDTH-1:0] systolic_update_w        ,
    //input [                      SUBARRAY_WIDTH-1:0] systolic_update_idx      ,
    //input [SUBARRAY_WIDTH*8-1:0] sdr_encoder_out  ,
    //input [SUBARRAY_WIDTH*8-1:0] sdr_encoder_sign_out  ,

    output      [                                   31:0] prdata           ,
    output                                                pready           ,
    // Weight shifter sram interface
    output wire [                           N_WGT_MX-1:0] wgt_sram_r_en    ,
    output wire [           WGT_SRAM_ADDR_W*N_WGT_MX-1:0] wgt_sram_raddr   ,
    input  wire [N_WGT_PER_MX*(WGT_WIDTH+1)*N_WGT_MX-1:0] wgt_sram_rdata   ,
    // XIN sram interface
    output wire [                           N_XIN_MX-1:0] xin_sram_r_en    ,
    output wire [           XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_sram_raddr   ,
    input  wire [                       8*8*N_XIN_MX-1:0] xin_sram_rdata   ,

    output      [                 16*SUBARRAY_HEIGHT-1:0] act_block_results
);
    genvar i, j;

    function [31:0] clog2 (input [31:0] x);
        reg [31:0] x_tmp;
        begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
        end
    endfunction

    wire [ 1:0] turn_on_signal       ;
    wire [31:0] systolic_width_size  ;
    wire [31:0] systolic_height_size ;
    wire [31:0] input_xin_width_size ;
    wire [31:0] input_xin_height_size;
    wire [4:0] wgt_budget           ;
    wire        data_budget          ;


    wire [31:0] reg_write_data;
    wire [15:0] reg_addr      ;
    wire [31:0] reg_read_data ;
    wire        reg_write     ;
    wire        reg_read      ;
    wire        reg_idle      ;

    // access the reg file with the axi2apb interface

    apb2reg i_apb2reg (
        .clk           (clk           ),
        .reset_n       (reset_n       ),
        .psel          (psel          ),
        .paddr         (paddr[15:2]   ),
        .pwrite        (pwrite        ),
        .pwdata        (pwdata        ),
        .penable       (penable       ),
        .prdata        (prdata        ),
        .pready        (pready        ),
        .reg_write_data(reg_write_data),
        .reg_addr      (reg_addr      ),
        .reg_read_data (reg_read_data ),
        .reg_write     (reg_write     ),
        .reg_read      (reg_read      ),
        .reg_idle      (reg_idle      )
    );

    // save the signal to the reg file
    reg_define i_reg_define (
        .turn_on_signal(turn_on_signal          ),
        .budget        ({data_budget,wgt_budget}),
        .systolic_width_size (systolic_width_size),
        .systolic_height_size (systolic_height_size),
        .input_xin_width_size (input_xin_width_size),
        .input_xin_height_size (input_xin_height_size),
        .write_data    (reg_write_data          ),
        .addr          (reg_addr                ),
        .read_data     (reg_read_data           ),
        .write         (reg_write               ),
        .read          (reg_read                ),
        .clk           (clk                     ),
        .reset_n       (reset_n)
    );

    
    // Activation input or feature map input
    wire [                N_XIN_MX-1:0] xin_shift_start    ;
    wire [XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_start_addr     ;
    wire [         XIN_SRAM_ADDR_W-1:0] xin_img_width_size ;
    wire [         XIN_SRAM_ADDR_W-1:0] xin_img_height_size;
    wire [            8*8*N_XIN_MX-1:0] xin_serial_output  ; // the first 8 is the data bitwidth, the second 8 is the number columns per data memory
    wire [              8*N_XIN_MX-1:0] xin_serial_en      ;

    generate
        for (i = 0; i < N_XIN_MX; i=i+1) begin: g_xin_shifter
            data_memory #(.SRAM_DEPTH(XIN_SRAM_DEPTH)) i_j_shifter_MX_cell (
                .clk            (clk                                                              ),
                .reset_n        (reset_n                                                          ),
                .sram_en        (xin_sram_r_en        [i]                                         ),
                .sram_addr      (xin_sram_raddr       [i*XIN_SRAM_ADDR_W   +: XIN_SRAM_ADDR_W  ]  ),
                .sram_data      (xin_sram_rdata[i*64                 +: 64                ]       ),
                .start          (xin_shift_start      [i]                                         ),
                .start_addr     ('b0   ),
                .img_width_size (xin_img_width_size                                               ),
                .img_height_size(xin_img_height_size                                              ),
                .data_output    (xin_serial_output    [i*64                 +: 64                ]),
                .data_en        (xin_serial_en        [i               ]                          )
            );
            assign xin_shift_start[i] = turn_on_signal[1];
        end

    endgenerate

    assign xin_img_width_size  = input_xin_width_size [XIN_SRAM_ADDR_W-1:0];
    assign xin_img_height_size = input_xin_height_size[XIN_SRAM_ADDR_W-1:0];
    

    
    //////////////////////////////////////////////////////
    //                    SDR Encoder
    //////////////////////////////////////////////////////

    wire [SUBARRAY_WIDTH*8-1:0] sdr_encoder_out  ;
    wire [SUBARRAY_WIDTH*8-1:0] sdr_encoder_sign_out  ;
    wire [SUBARRAY_WIDTH-1:0] sdr_encoder_start;
    generate
        for (i = 0; i < SUBARRAY_WIDTH; i=i+1) begin: g_sdr_encoder
            sdr_encoder_MX sdr_MX_cell (
                .clk          (clk                              ),
                .reset        (~reset_n                         ),
                .input_stream (xin_serial_output        [i*8+:8]),
                .enable       (sdr_encoder_start       [i]      ),
                .budget       (data_budget                      ),
                .output_stream(sdr_encoder_out      [i*8 +:8 ]  ),
                .sign_stream  (sdr_encoder_sign_out  [i*8 +:8 ] )
            );
            assign sdr_encoder_start[i] = turn_on_signal[1];
        end

    endgenerate
     

    //////////////////////////////////////////////////////
    //                    Term quantizer
    //////////////////////////////////////////////////////

    wire [SUBARRAY_WIDTH*8*DATA_WIDTH-1:0] term_quantizer_out;
    wire [SUBARRAY_WIDTH*8-1:0] term_quantizer_sign_out;
    generate
        for (i = 0; i < SUBARRAY_HEIGHT*8; i=i+1) begin: g_term_quantizer
            term_quantizer term_quantized_cell (
                .clk              (clk                                                 ),
                .reset            (~reset_n                                            ),
                .input_stream     (sdr_encoder_out        [i]                          ),
                .input_sign_stream(sdr_encoder_sign_out        [i]                     ),
                .output_s         (term_quantizer_out      [i*DATA_WIDTH +:DATA_WIDTH ]),
                .output_sign_s    (term_quantizer_sign_out      [i ]                   )
            );
        end

    endgenerate
   

    
    // Weight input
    // N_WGT_PER_MX: number of weight sram, N_WGT_MX: number of systolic column per weight sram, WGT_WIDTH+1+4
    wire [N_WGT_PER_MX*(WGT_WIDTH+1)*N_WGT_MX*GROUP_SIZE-1:0] wgt_serial_output;
    wire [                                  N_WGT_PER_MX-1:0] wgt_serial_en    ;
    wire [                               WGT_SRAM_ADDR_W-1:0] wgt_img_size     ;
    wire [                                  N_WGT_PER_MX-1:0] wgt_shift_start  ;

    // N_WGT_MX is 8, each weight buffer controls 8 columns of systolic array
    generate
        for (i = 0; i < N_WGT_MX; i=i+1) begin: g_wgt_shifter
            weight_memory #(
                .SRAM_DEPTH (WGT_SRAM_DEPTH )
            ) i_j_wgt_shifter_MX_cell (
                .clk        (clk                                                                                                      ),
                .reset_n    (reset_n                                                                                                  ),
                .sram_en    (wgt_sram_r_en     [i*1                 +: 1                ]                                             ),
                .sram_addr  (wgt_sram_raddr    [i*1*WGT_SRAM_ADDR_W +: 1*WGT_SRAM_ADDR_W]                                             ),
                .sram_data  (wgt_sram_rdata    [i*N_WGT_PER_MX*(WGT_WIDTH+1)               +: N_WGT_PER_MX*(WGT_WIDTH+1)             ]), // each weight buffer store the weight exponents for 8 columns
                .start      (wgt_shift_start   [i]                                                                                    ),
                .end_addr   (wgt_img_size                                                                                             ), // specify the ending address
                .wgt_budget (wgt_budget                                                                                               ),
                .data_output(wgt_serial_output [i*N_WGT_PER_MX*(WGT_WIDTH+1) +: N_WGT_PER_MX*(WGT_WIDTH+1)]                           ),
                .data_en    (wgt_serial_en     [i              ]                                                                      )
            );

            assign wgt_shift_start[i] = turn_on_signal[0];
        end

    endgenerate

    assign wgt_img_size = systolic_height_size[WGT_SRAM_ADDR_W-1:0];
    //assign sytolic_array_idle[0] = (&wgt_shift_idle);
    

    //////////////////////////////////////////////////////
    // Systolic main
    //////////////////////////////////////////////////////
    
    wire [                     SUBARRAY_HEIGHT-1:0] systolic_mac_en          ;
    wire [(WGT_WIDTH+1)*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_wgt ;
    wire [DATA_WIDTH*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in_xin ;
    wire [WGT_WIDTH*GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_in     ;
    wire [           GROUP_SIZE*SUBARRAY_WIDTH-1:0] systolic_dataflow_sign_in;
    wire [                      SUBARRAY_WIDTH-1:0] systolic_update_w        ;
    //wire [                      SUBARRAY_WIDTH-1:0] systolic_update_idx      ;
    wire [                  16*SUBARRAY_HEIGHT-1:0] systolic_result          ;
    wire [                  16*SUBARRAY_HEIGHT-1:0] systolic_result_en       ;
    

    j_systolic_array #(
        .SUBARRAY_WIDTH (SUBARRAY_WIDTH ),
        .SUBARRAY_HEIGHT(SUBARRAY_HEIGHT),
        .GROUP_SIZE     (GROUP_SIZE     ),
        .DATA_WIDTH     (DATA_WIDTH     ),
        .WGT_WIDTH      (WGT_WIDTH      ),
        .MAX_BUDGET     (MAX_BUDGET     )
    ) i_j_systolic_array (
        .clk                (clk                      ),
        .reset              (~reset_n                 ),
        .accumulation_in_pos('b0                      ),
        .accumulation_in_neg('b0                      ),
        .mac_en             (systolic_mac_en          ),
        .dataflow_in        (systolic_dataflow_in     ),
        .dataflow_sign_in   (systolic_dataflow_sign_in),
        .update_w           (systolic_update_w        ),
        .update_idx         (systolic_update_w      ),
        .result             (systolic_result          ),
        .result_en          (systolic_result_en       )
    );


    //////////////////////////////////////////////////////
    // Acitvation Block
    //////////////////////////////////////////////////////
    //wire [                  16*SUBARRAY_HEIGHT-1:0] act_block_results          ;

    generate
        for (i = 0; i < SUBARRAY_HEIGHT; i=i+1) begin: g_activation_block
            act_block act_block_cell (
                .en       (systolic_result_en[i]             ),
                .input_act(systolic_result    [i*16 +: 16 ]  ),
                .result   (act_block_results     [i*16 +: 16])
            );
        end
    endgenerate

    generate
        for (j = 0; j < SUBARRAY_WIDTH; j=j+1) begin
            assign systolic_update_w[j]                                                           = wgt_serial_en[j];
            //assign systolic_update_idx[j]                                                         = wgt_serial_en[j];
            assign systolic_dataflow_in_wgt[j*(WGT_WIDTH+1)*GROUP_SIZE+:(WGT_WIDTH+1)*GROUP_SIZE] = systolic_update_w[j] ? wgt_serial_output[j*(WGT_WIDTH+1)*GROUP_SIZE+:(WGT_WIDTH+1)*GROUP_SIZE]: 'b0;
        end


        for (j = 0; j < SUBARRAY_WIDTH; j=j+1) begin
            assign systolic_dataflow_in_xin[j*(DATA_WIDTH)*GROUP_SIZE+:(DATA_WIDTH)*GROUP_SIZE] = ~systolic_update_w[j] ? term_quantizer_out[j*DATA_WIDTH*GROUP_SIZE+:DATA_WIDTH*GROUP_SIZE]: 1'b0;
            assign systolic_dataflow_sign_in[j*1*GROUP_SIZE+:1*GROUP_SIZE] = ~systolic_update_w[j] ? term_quantizer_sign_out[j*1*GROUP_SIZE+:1*GROUP_SIZE]: 1'b0;

        end

    endgenerate

    assign systolic_dataflow_in = systolic_dataflow_in_wgt | systolic_dataflow_in_xin;
    
    generate
        for (j = 0; j < SUBARRAY_HEIGHT; j=j+1) begin
            assign systolic_mac_en[j] = turn_on_signal[1];
        end
    endgenerate
    


    // idle
    //assign sytolic_array_idle[1] = (&deacc_shift_idle) & (&acc_shift_idle) & (&xin_shift_idle);

endmodule


