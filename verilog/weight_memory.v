
module weight_memory #(
	parameter SRAM_DEPTH  = 256*256*4        ,
	parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
	input                         clk        ,
	input                         reset_n    ,
	output reg                    sram_en    ,
	output reg  [SRAM_ADDR_W-1:0] sram_addr  ,
	input       [        4*8-1:0] sram_data  ,
	input                         start      ,
	input  wire [SRAM_ADDR_W-1:0] end_addr   ,
	input       [            4:0] wgt_budget , //from 0 to 31
	output wire [        4*8-1:0] data_output,
	output wire                   data_en
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

	reg [        4*8-1:0] data_output_reg;
	reg [SRAM_ADDR_W-1:0] counter        ;
	reg [            4:0] counter_budget ;
	assign data_output = data_output_reg;
	assign data_en     = (reset_n) && (counter <= end_addr);

	always @ (posedge clk) begin
		if (~reset_n) begin
			sram_addr       <= 'b0;
			sram_en         <= 'b0;
			data_output_reg <= 'b0;
			counter         <= 'b0;
			counter_budget  <= 'b0;
		end
		else if (start) begin

			counter_budget     <= ((counter_budget + 1) && (counter_budget < wgt_budget)) | ('b0 && (counter_budget >= wgt_budget));
			sram_en         <= 1;
			sram_addr       <= ((counter + counter_budget) && (counter_budget < wgt_budget)) | ((counter + 5'b11111) && (counter_budget >= wgt_budget));
			counter         <= (counter && (counter_budget < wgt_budget))|((counter + 5'b11111) && (counter_budget >= wgt_budget));
			data_output_reg <= sram_data;
		end
	end


endmodule      
