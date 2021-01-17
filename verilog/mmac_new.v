

module mmac_new #(
	parameter GROUP_SIZE  = 16                    ,
	parameter DATA_WIDTH  = 3                     ,
	parameter WGT_WIDTH   = 3                     ,
	parameter MAX_BUDGET  = 16                    ,
	parameter IDX_WIDTH   = clog2(GROUP_SIZE)     ,
	parameter WGT_IDX_LEN = MAX_BUDGET * IDX_WIDTH
) (
	input                                  clk          ,
	input                                  reset        ,
	input      [                     15:0] pos_acc      ,
	input      [                     15:0] neg_acc      ,
	input                                  update_w_i   ,
	input                                  update_idx_i ,
	input                                  mac_en_i     ,
	input      [DATA_WIDTH*GROUP_SIZE-1:0] data_in      ,
	input      [           GROUP_SIZE-1:0] data_sign_in ,
	output reg                             update_w_o   ,
	output reg                             update_idx_o ,
	output reg                             mac_en_o     ,
	output reg [DATA_WIDTH*GROUP_SIZE-1:0] data_out     ,
	output reg [           GROUP_SIZE-1:0] data_sign_out,
	output reg [                     15:0] out_pos      ,
	output reg [                     15:0] out_neg
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


	reg [WGT_WIDTH*MAX_BUDGET-1:0] W     ;
	reg [          MAX_BUDGET-1:0] W_sign;
	reg [IDX_WIDTH*MAX_BUDGET-1:0] W_idx ;
	reg [  WGT_WIDTH-1:0] wgt_in_reg     ;
    reg [          MAX_BUDGET-1:0] wgt_sign_in_reg;
    reg [IDX_WIDTH-1:0] wgt_idx_in_reg ;
    wire [  WGT_WIDTH-1:0] wgt_in     ;
    wire [          0:0] wgt_sign_in;
    wire [IDX_WIDTH-1:0] wgt_idx_in ;
    
	// update the weight, weight sign and weight index

	always @(posedge clk) begin : proc_W
		if(reset) begin
			W      <= 'b0;
			W_sign <= 'b0;
			W_idx  <= 'b0;
			wgt_in_reg      <= 'b0;
			wgt_sign_in_reg <= 'b0;
			wgt_idx_in_reg  <= 'b0;
		end else if(update_w_i) begin
			W      <= data_in;
			W_sign <= data_sign_in;
		end else if(update_idx_i) begin
			W_idx <= {data_in, data_sign_in};
		end else if(~update_w_i && ~update_idx_i && mac_en_i) begin
			wgt_in_reg      <= W[WGT_WIDTH*MAX_BUDGET-1:WGT_WIDTH*(MAX_BUDGET-1)];
			W               <= {W[WGT_WIDTH*(MAX_BUDGET-1)-1:0], {(WGT_WIDTH){1'b0}}};
			wgt_sign_in_reg <= W_sign[MAX_BUDGET-1];
			W_sign          <= {W_sign[(MAX_BUDGET-2):0], 1'b0};
			wgt_idx_in_reg  <= W_idx[IDX_WIDTH*MAX_BUDGET-1:IDX_WIDTH*(MAX_BUDGET-1)];
			W_idx           <= {W_idx[IDX_WIDTH*(MAX_BUDGET-1)-1:0], W_idx[IDX_WIDTH*MAX_BUDGET-1:IDX_WIDTH*(MAX_BUDGET-1)]};
		end
	end

	// update the data
	wire [15:0] out_pos_net, out_neg_net;
	always @(posedge clk) begin : comp_W
		if(reset) begin
			data_out      <= 'b0;
			data_sign_out <= 'b0;
			out_pos       <= 'b0;
			out_neg       <= 'b0;
			update_w_o    <= 'b0;
			update_idx_o  <= 'b0;
			mac_en_o      <= 'b0;
		end else if (mac_en_i) begin
			data_out      <= data_in ;
			data_sign_out <= data_sign_in;
			out_pos       <= out_pos_net;
			out_neg       <= out_neg_net;
			update_w_o    <= update_w_i;
			update_idx_o  <= update_idx_i;
			mac_en_o      <= mac_en_i;
		end
	end




	// update the data, data sign
	reg  [          0:0] data_sign_in_reg;
	wire [  DATA_WIDTH-1:0] data_in_exp     ;
	wire [          0:0] data_sign_in_exp;
	wire [          0:0] exp_sign        ;
	wire [WGT_WIDTH:0] exp_summation   ;
	assign wgt_in      = wgt_in_reg;
	assign wgt_sign_in = wgt_sign_in_reg;
	assign wgt_idx_in  = wgt_idx_in_reg;

	assign exp_summation = wgt_in + data_in_exp;
	assign exp_sign      = wgt_sign_in ^ data_sign_in_exp;


	//mux16to1 #(.INPUT_WIDTH(DATA_WIDTH)) (.Out(data_in_exp), .Sel(wgt_idx_in), .In1(data_in[DATA_WIDTH-1:0]),.In2(data_in[2*DATA_WIDTH-1:DATA_WIDTH]),.In3(data_in[3*DATA_WIDTH-1:2*DATA_WIDTH]),.In4(data_in[4*DATA_WIDTH-1:3*DATA_WIDTH]),.In5(data_in[5*DATA_WIDTH-1:4*DATA_WIDTH]),.In6(data_in[6*DATA_WIDTH-1:5*DATA_WIDTH]),.In7(data_in[7*DATA_WIDTH-1:6*DATA_WIDTH]),.In8(data_in[8*DATA_WIDTH-1:7*DATA_WIDTH]),
	//	.In9(data_in[9*DATA_WIDTH-1:8*DATA_WIDTH]),.In10(data_in[10*DATA_WIDTH-1:9*DATA_WIDTH]),.In11(data_in[11*DATA_WIDTH-1:10*DATA_WIDTH]),.In12(data_in[12*DATA_WIDTH-1:11*DATA_WIDTH]),.In13(data_in[13*DATA_WIDTH-1:12*DATA_WIDTH]),.In14(data_in[14*DATA_WIDTH-1:13*DATA_WIDTH]),.In15(data_in[15*DATA_WIDTH-1:14*DATA_WIDTH]),.In16(data_in[16*DATA_WIDTH-1:15*DATA_WIDTH]));

	//mux16to1 #(.INPUT_WIDTH(1)) (.Out(data_sign_in_exp), .Sel(wgt_idx_in), .In1(data_sign_in[1]),.In2(data_sign_in[2]),.In3(data_sign_in[3]),.In4(data_sign_in[4]),.In5(data_sign_in[5]),.In6(data_sign_in[6]),.In7(data_sign_in[7]),.In8(data_sign_in[8]),
	//	.In9(data_sign_in[9]),.In10(data_sign_in[10]),.In11(data_sign_in[11]),.In12(data_sign_in[12]),.In13(data_sign_in[13]),.In14(data_sign_in[14]),.In15(data_sign_in[15]),.In16(data_sign_in[16]));

	mux8to1 #(.INPUT_WIDTH(DATA_WIDTH),.IDX_WIDTH(IDX_WIDTH)) (.Out(data_in_exp), .Sel(wgt_idx_in), .In1(data_in[DATA_WIDTH-1:0]),.In2(data_in[2*DATA_WIDTH-1:DATA_WIDTH]),.In3(data_in[3*DATA_WIDTH-1:2*DATA_WIDTH]),.In4(data_in[4*DATA_WIDTH-1:3*DATA_WIDTH]),.In5(data_in[5*DATA_WIDTH-1:4*DATA_WIDTH]),.In6(data_in[6*DATA_WIDTH-1:5*DATA_WIDTH]),.In7(data_in[7*DATA_WIDTH-1:6*DATA_WIDTH]));

	mux8to1 #(.INPUT_WIDTH(1),.IDX_WIDTH(IDX_WIDTH)) (.Out(data_sign_in_exp), .Sel(wgt_idx_in), .In1(data_sign_in[1]),.In2(data_sign_in[2]),.In3(data_sign_in[3]),.In4(data_sign_in[4]),.In5(data_sign_in[5]),.In6(data_sign_in[6]),.In7(data_sign_in[7]));

	term_accumulator #(.ACC_BIT_WIDTH(16),.SHIFT_WIDTH(WGT_WIDTH+1)) (.clk(clk), .reset(reset), .en(mac_en_i),.pos_acc(pos_acc),.neg_acc(neg_acc),.sign(exp_sign),.shift_pos(exp_summation),.pos_result(out_pos_net),.neg_result(out_neg_net));
endmodule
