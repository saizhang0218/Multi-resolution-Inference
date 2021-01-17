`timescale 1ns / 1ps


module act_block (
	input  [                   0:0] en   ,
	input  [                16-1:0] input_act,
	output [                16-1:0] result  
);
    wire [16:0] intermediate_result1,intermediate_result2;
    assign intermediate_result1 = (input_act) & {(16){en}};
    assign intermediate_result2 = (intermediate_result1>0) ? intermediate_result1 : 16'b0;
    assign result = intermediate_result2;
endmodule
