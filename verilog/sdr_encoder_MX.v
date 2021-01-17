`timescale 1ns / 1ps


module sdr_encoder_MX (
	input  clk          ,
	input  reset        ,
	input  [7:0] input_stream ,
	input  enable       ,
	input  budget       , // 0 means the budget is 2, 1 means the budget is 3
	output [7:0] output_stream,
	output [7:0] sign_stream
);

    reg  [2:0] counter;
    reg  [8*8-1:0] input_stream_dummy_reg;
	always @(posedge clk) begin
		if(reset) begin
			{input_stream_dummy_reg, counter} <= 'b0;
		end else begin
			counter <= counter + 1;
			input_stream_dummy_reg[counter*8+:8] <= input_stream;
		end
	end
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[7:0]),.enable(enable),.budget(budget),.output_stream(output_stream[0]),.sign_stream(sign_stream[0]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[15:8]),.enable(enable),.budget(budget),.output_stream(output_stream[1]),.sign_stream(sign_stream[1]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[23:16]),.enable(enable),.budget(budget),.output_stream(output_stream[2]),.sign_stream(sign_stream[2]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[31:24]),.enable(enable),.budget(budget),.output_stream(output_stream[3]),.sign_stream(sign_stream[3]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[39:32]),.enable(enable),.budget(budget),.output_stream(output_stream[4]),.sign_stream(sign_stream[4]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[47:40]),.enable(enable),.budget(budget),.output_stream(output_stream[5]),.sign_stream(sign_stream[5]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[55:48]),.enable(enable),.budget(budget),.output_stream(output_stream[6]),.sign_stream(sign_stream[6]));
	sdr_encoder (.clk(clk), .reset(reset), .input_stream(input_stream_dummy_reg[63:56]),.enable(enable),.budget(budget),.output_stream(output_stream[7]),.sign_stream(sign_stream[7]));


endmodule