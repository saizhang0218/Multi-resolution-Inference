`timescale 1ns / 1ps


module term_quantizer (
input            clk              ,
input            reset            ,
input            input_stream     ,
input            input_sign_stream,
output reg [2:0] output_s,
output reg       output_sign_s
);

reg [1:0] counter_total;
reg [2:0] output_s1, output_s2;
reg       output_sign1, output_sign2;

always @(posedge clk) begin
if(reset) begin
{output_s1,output_s2}    <= 'b0;
{output_sign1,output_sign2} <= 'b0;
counter_total <= 'b0;
end else begin
counter_total <= counter_total + 1'b1;
output_s1     <= counter_total * input_stream;  // convert into power of 2 exponent
output_s2     <= output_s1;
output_s     <= output_s2;
output_sign1  <= input_sign_stream;
output_sign2  <= output_sign1;
output_sign_s  <= output_sign2;
end
end


endmodule