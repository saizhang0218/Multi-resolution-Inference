`timescale 1ns / 1ps


module term_accumulator #(
  parameter ACC_BIT_WIDTH = 16,
  parameter SHIFT_WIDTH   = 3
) (
  input                          clk       ,
  input                          reset     ,
  input                          en        ,
  input      [ACC_BIT_WIDTH-1:0] pos_acc   ,
  input      [ACC_BIT_WIDTH-1:0] neg_acc   ,
  input      [              0:0] sign      ,
  input      [SHIFT_WIDTH-1:0] shift_pos ,
  output reg [ACC_BIT_WIDTH-1:0] pos_result,
  output reg [ACC_BIT_WIDTH-1:0] neg_result
);


  always @ (posedge clk) begin
    if (reset) begin
      pos_result <= 'b0;
      neg_result <= 'b0;
    end
    else
      begin
        pos_result <= pos_acc + ((1'b1>>shift_pos) & {(ACC_BIT_WIDTH){sign}}) & {(ACC_BIT_WIDTH){en}};
        neg_result <= neg_acc + ((1'b1>>shift_pos) & {(ACC_BIT_WIDTH){~sign}}) & {(ACC_BIT_WIDTH){en}};
      end
  end
endmodule
