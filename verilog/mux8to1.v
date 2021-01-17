`timescale 1ns / 1ps

module mux8to1 #(
  parameter INPUT_WIDTH = 4,
  parameter IDX_WIDTH = 3
) (
  Out,
  Sel,
  In1,
  In2,
  In3,
  In4,
  In5,
  In6,
  In7
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

  input  [INPUT_WIDTH-1:0] In1, In2, In3, In4, In5, In6, In7;
  input  [            2:0] Sel;
  output [INPUT_WIDTH-1:0] Out;
  reg [INPUT_WIDTH-1:0] Out;


//Check the state of the input lines

  always @ (In1 or In2 or In3 or In4 or In5 or In6 or In7 or Sel)
    begin
      case (Sel)
        3'b000 : Out <= In1;
        3'b001 : Out <= In2;
        3'b010 : Out <= In3;
        3'b011 : Out <= In4;
        3'b100 : Out <= In5;
        3'b101 : Out <= In6;
        3'b110 : Out <= In7;

      endcase

    end

endmodule