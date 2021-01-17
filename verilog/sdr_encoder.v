`timescale 1ns / 1ps

module sdr_encoder (
	input  clk          ,
	input  reset        ,
	input  [7:0] input_stream ,
	input  enable       ,
	input  budget       , // 0 means the budget is 2, 1 means the budget is 3
	output output_stream,
	output sign_stream
);

	wire [1:0] dummy, current_budget        ;
	reg        input_stream_reg, input_stream_reg_delay, output_stream_reg, sign_stream_reg, input_stream_dummy_reg, status;
    reg  [1:0] current_budget_reg;
    reg  [2:0] counter;
    wire [2:0] counter_wire;
    assign counter_wire = counter;
	always @(posedge clk) begin
		if(reset) begin
			{input_stream_reg,input_stream_reg_delay,input_stream_dummy_reg, counter} <= 'b0;
		end else begin
			counter <= counter + 1;
			input_stream_dummy_reg <= input_stream[counter];
			input_stream_reg       <= input_stream_dummy_reg;
			input_stream_reg_delay <= input_stream_reg;
		end
	end

	assign dummy[1]      = input_stream_reg_delay;
	assign dummy[0]      = input_stream_reg;
	assign sign_stream   = (sign_stream_reg & (enable)) & (current_budget< ((2'b10) && {(2){budget}} | (2'b11) && {(2){~budget}})) ;
    assign output_stream  = (output_stream_reg & (enable)) & (current_budget< ((2'b10) && {(2){budget}} | (2'b11) && {(2){~budget}})) ;
	assign current_budget = current_budget_reg;
	
	always @ (posedge clk) begin
		if(reset) begin
			{current_budget_reg,output_stream_reg,sign_stream_reg, status} <= 'b0;
		end
		else begin
			current_budget_reg <= current_budget_reg + (output_stream==1'b1);
			case ({status,dummy[0],dummy[1]})
				3'b011 : begin
					output_stream_reg <= 'b1;
					sign_stream_reg   <= 'b1;
					status            <= 'b1;
				end
				3'b100 : begin
					output_stream_reg <= 'b1;
					sign_stream_reg   <= 'b0;
					status            <= 'b0;
				end
				3'b101 : begin
					output_stream_reg <= 'b0;
					sign_stream_reg   <= 'b0;
					status            <= 'b1;
				end
				3'b110 : begin
					output_stream_reg <= 'b1;
					sign_stream_reg   <= 'b1;
					status            <= 'b1;
				end
				3'b111 : begin
					output_stream_reg <= 'b0;
					sign_stream_reg   <= 'b0;
					status            <= 'b1;
				end
				default : begin
					output_stream_reg <= dummy[0];
					sign_stream_reg   <= 'b0;
					status            <= 'b0;
				end
			endcase
		end
	end
endmodule