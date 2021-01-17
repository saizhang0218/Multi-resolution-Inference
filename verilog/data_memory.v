module data_memory #(
	parameter SRAM_DEPTH  = 256*256*4        ,
	parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
	input                         clk            ,
	input                         reset_n        ,
	output reg                    sram_en        ,
	output reg  [SRAM_ADDR_W-1:0] sram_addr      ,
	input       [        8*8-1:0] sram_data      , // {data[7:0]}
	input                         start          ,
	input  wire [SRAM_ADDR_W-1:0] start_addr     ,
	input  wire [SRAM_ADDR_W-1:0] img_width_size ,
	input  wire [SRAM_ADDR_W-1:0] img_height_size,
	output wire [        8*8-1:0] data_output    ,
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

	reg [        8*8-1:0] data_output_reg;
	reg [SRAM_ADDR_W-1:0] counter        ;
	assign data_output = data_output_reg;
	assign data_en     = (reset_n) && (counter <= (start_addr + img_width_size * img_height_size));

	always @ (posedge clk) begin
		if (~reset_n) begin
			sram_addr       <= 'b0;
			sram_en         <= 'b0;
			data_output_reg <= 'b0;
			counter         <= start_addr;
		end
		else if (start) begin
			counter         <= counter + 1;
			sram_en         <= 1;
			sram_addr       <= counter;
			data_output_reg <= sram_data;
		end
	end


endmodule      
