#define _USE_MATH_DEFINES
#include <cmath>
#include <fftw3.h>
#include <stddef.h>
#include "complex_array_class.h"

#define sqr(x) ((x)*(x))

//FFTクラス実装

// //fftw3はDLLをプロジェクトのフォルダに置いて動的リンクライブラリ（暗黙的リンク）として
// //.defがあるので明示的リンクも可？
// #pragma comment(lib, "libfftw3-3.lib")
// #pragma comment(lib, "libfftw3f-3.lib")
// #pragma comment(lib, "libfftw3l-3.lib")


//inline
//最大値
template <class Type>
inline Type get_max(Type* data, int size)
{
	Type max = data[0];
	for (int i = 0; i < size; i++) {
		if (max < data[i]) {
			max = data[i];
		}
	}

	return max;
}

//最小値
template <class Type>
inline Type get_min(Type* data, int size)
{
	Type min = data[0];
	for (int i = 0; i < size; i++) {
		if (min > data[i]) {
			min = data[i];
		}
	}

	return min;
}

//最大値で正規化
template <class Type>
inline void norm_max(Type* data_in, int size, double* data_out)
{
	Type max;
	max = get_max(data_in, size);

	for (int i = 0; i < size; i++) {
		data_out[i] = (double)((double)data_in[i] / (double)max);
	}

}

//最大値と最小値で正規化(確実に0~1)
template <class Type>
inline void norm_max_min(Type* data_in, int size, double* data_out)
{
	Type max;
	max = get_max(data_in, size);
	Type min;
	min = get_min(data_in, size);

	double* tmp;
	tmp = new double[size];

	for (int i = 0; i < size; i++) {
		tmp[i] = (double)((data_in[i] - min) / (max - min));
	}

	for (int i = 0; i < size; i++) {
		data_out[i] = tmp[i];
	}


	delete[]tmp;
}






My_Fft::My_Fft(int sx, int sy)
{
	x = sx;
	y = sy;
	Re_in = new double[x * y];
	Im_in = new double[x * y];
	Re_out = new double[x * y];
	Im_out = new double[x * y];

}


void My_Fft::data_to_in(double* Re, double* Im)
{
	for (int i = 0; i < x * y; i++) {
		Re_in[i] = Re[i];
		Im_in[i] = Im[i];
	}
}


void My_Fft::out_to_data(double* Re, double* Im)
{
	for (int i = 0; i < x * y; i++) {
		Re[i] = Re_out[i];
		Im[i] = Im_out[i];
	}
}



void My_Fft::fft2d()
{
	fftw_complex* in, * out;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	out = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			in[i * x + j][0] = Re_in[i * x + j];
			in[i * x + j][1] = Im_in[i * x + j];
		}
	}

	p = fftw_plan_dft_2d(y, x, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	//正規化
	for (int i = 0; i < y * x; i++) {
		out[i][0] /= (double)(y * x);
		out[i][1] /= (double)(y * x);

	}


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re_out[i * x + j] = out[i * x + j][0];
			Im_out[i * x + j] = out[i * x + j][1];
		}
	}

	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
}


void My_Fft::ifft2d()
{
	fftw_complex* in, * out;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	out = (fftw_complex*)fftw_malloc((size_t)y * (size_t)x * sizeof(fftw_complex));
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			in[i * x + j][0] = Re_in[i * x + j];
			in[i * x + j][1] = Im_in[i * x + j];
		}
	}
	p = fftw_plan_dft_2d(y, x, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Re_out[i * x + j] = out[i * x + j][0];
			Im_out[i * x + j] = out[i * x + j][1];
		}
	}
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
}



My_Fft::~My_Fft()
{
	delete[]Re_in;
	delete[]Im_in;
	delete[]Re_out;
	delete[]Im_out;

}




//複素配列クラス実装

My_Complex_Array::My_Complex_Array(int s)
	:s(s)
{
	//this->s = s;
	
	Re = new double[this->s];
	Im = new double[this->s];

	//初期化
	for (int i = 0; i < this->s; i++) {
		Re[i] = 0;
		Im[i] = 0;
	}
}


My_Complex_Array::~My_Complex_Array()
{
	delete[]Re;
	delete[]Im;
}



//実部、虚部両方格納
//doubleを格納
void My_Complex_Array::data_to_ReIm(double* Rein, double* Imin)
{
	for (int i = 0; i < s; i++) {
		Re[i] = Rein[i];
		Im[i] = Imin[i];
	}
}

//intを格納
void My_Complex_Array::data_to_ReIm(int* Rein, int* Imin)
{
	for (int i = 0; i < s; i++) {
		Re[i] = (double)Rein[i];
		Im[i] = (double)Imin[i];

	}
}

//unsigned charを格納
void My_Complex_Array::data_to_ReIm(unsigned char* Rein, unsigned char* Imin)
{
	for (int i = 0; i < s; i++) {

		long tmp = (long)Rein[i];
		Re[i] = (double)tmp;

		tmp = (long)Imin[i];
		Im[i] = (double)tmp;


	}
}


//実部のみ格納
void My_Complex_Array::data_to_ReIm(double* Rein)
{
	for (int i = 0; i < s; i++) {
		Re[i] = Rein[i];
	}
}

void My_Complex_Array::data_to_ReIm(int* Rein)
{
	for (int i = 0; i < s; i++) {
		Re[i] = (double)Rein[i];

	}
}

void My_Complex_Array::data_to_ReIm(unsigned char* Rein)
{
	for (int i = 0; i < s; i++) {

		long tmp = (long)Rein[i];
		Re[i] = (double)tmp;

	}
}





//振幅出力
void My_Complex_Array::power(double* pow)
{
	for (int i = 0; i < s; i++) {
		pow[i] = sqrt(sqr(Re[i]) + sqr(Im[i]));
	}
}




//複素乗算
void My_Complex_Array::mul_complex(My_Complex_Array* opponent, My_Complex_Array* out)
{
	double* Retmp, * Imtmp;
	Retmp = new double[s];
	Imtmp = new double[s];

	for (int i = 0; i < s; i++) {
		Retmp[i] = (Re[i] * opponent->Re[i]) - (Im[i] * opponent->Im[i]);
		Imtmp[i] = (Re[i] * opponent->Im[i]) + (Im[i] * opponent->Re[i]);

	}

	for (int i = 0; i < s; i++) {

		out->Re[i] = Retmp[i];
		out->Im[i] = Imtmp[i];

	}
	delete[]Retmp;
	delete[]Imtmp;
}

//格納されている複素配列と指定した複素配列オブジェクトの乗算結果を、このオブジェクトに格納
void My_Complex_Array::mul_complex(My_Complex_Array* opponent)
{
	double* Retmp, * Imtmp;
	Retmp = new double[s];
	Imtmp = new double[s];

	for (int i = 0; i < s; i++) {
		Retmp[i] = (Re[i] * opponent->Re[i]) - (Im[i] * opponent->Im[i]);
		Imtmp[i] = (Re[i] * opponent->Im[i]) + (Im[i] * opponent->Re[i]);

	}
	for (int i = 0; i < s; i++) {

		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];

	}

	delete[]Retmp;
	delete[]Imtmp;
}

//格納されている複素配列と指定した複素配列の乗算結果を、このオブジェクトに格納
void My_Complex_Array::mul_complex(double* Re2, double* Im2)
{
	double* Retmp, * Imtmp;
	Retmp = new double[s];
	Imtmp = new double[s];

	for (int i = 0; i < s; i++) {
		
		Retmp[i] = Re[i] * Re2[i] - Im[i] * Im2[i];
		
		Imtmp[i] = Re[i] * Im2[i] + Im[i] * Re2[i];
		
	}

	for (int i = 0; i < s; i++) {
		
		Re[i] = Retmp[i];
		Im[i] = Imtmp[i];

	}

	delete[]Retmp;
	delete[]Imtmp;
}





//振幅(実部)情報inを正規化後、位相情報に変換
void My_Complex_Array::to_phase(double* in)
{
	double* tmp;
	tmp = new double[s];

	norm_max_min<double>(in, s, tmp);

	for (int i = 0; i < s; i++) {
		Re[i] = cos(2 * M_PI * tmp[i]);
		Im[i] = sin(2 * M_PI * tmp[i]);
	}

	delete[]tmp;
}

//振幅(実部)情報inを255で割って、位相情報に変換
void My_Complex_Array::to256_phase(double* in)
{
	for (int i = 0; i < s; i++) {
		Re[i] = cos(2 * M_PI * in[i] / 255);
		Im[i] = sin(2 * M_PI * in[i] / 255);
	}
}




//0埋めして拡大、ins = inx * iny、outx > inx、outy > iny
void My_Complex_Array::zeropad(My_Complex_Array* out,int outx, int outy, int inx, int iny)
{
	int* sx, * sy;
	sx = new int;
	sy = new int;

	*sx = (int)((outx - inx) / 2);
	*sy = (int)((outy - iny) / 2);

	for (int i = *sy; i < (*sy) + iny; i++) {
		for (int j = *sx; j < (*sx) + inx; j++) {

			out->Re[i * outx + j] = this->Re[(i - (*sy)) * inx + (j - (*sx))];
			out->Im[i * outx + j] = this->Im[(i - (*sy)) * inx + (j - (*sx))];
		}
	}

	delete sx;
	delete sy;
}

//中心を取り出して縮小、ins = inx * iny、outx < inx、outy < iny
void My_Complex_Array::extract_center(My_Complex_Array* out, int outx, int outy, int inx, int iny)
{
	int* sx, * sy;
	sx = new int;
	sy = new int;

	*sx = (int)((inx - outx) / 2);
	*sy = (int)((iny - outy) / 2);


	for (int i = *sy; i < (*sy) + outy; i++) {
		for (int j = *sx; j < (*sx) + outx; j++) {

			out->Re[(i - (*sy)) * outx + (j - (*sx))] = this->Re[i * inx + j];
			out->Im[(i - (*sy)) * outx + (j - (*sx))] = this->Im[i * inx + j];
		}
	}


	delete sx;
	delete sy;
}





//------------------------------------------2D--------------------------------------------



//0埋めして拡大、ins = inx * iny、outx > inx、outy > iny
void My_ComArray_2D::zeropad(My_ComArray_2D* out)
{
	int* sx, * sy;
	sx = new int;
	sy = new int;

	*sx = (int)((out->x - this->x) / 2);
	*sy = (int)((out->y - this->y) / 2);

	for (int i = *sy; i < (*sy) + this->y; i++) {
		for (int j = *sx; j < (*sx) + this->x; j++) {

			out->Re[i * out->x + j] = this->Re[(i - (*sy)) * this->x + (j - (*sx))];
			out->Im[i * out->x + j] = this->Im[(i - (*sy)) * this->x + (j - (*sx))];
		}
	}

	delete sx;
	delete sy;
}

//中心を取り出して縮小、ins = inx * iny、outx < inx、outy < iny
void My_ComArray_2D::extract_center(My_ComArray_2D* out)
{
	int* sx, * sy;
	sx = new int;
	sy = new int;

	*sx = (int)((this->x - out->x) / 2);
	*sy = (int)((this->y - out->y) / 2);


	for (int i = *sy; i < (*sy) + out->y; i++) {
		for (int j = *sx; j < (*sx) + out->x; j++) {

			out->Re[(i - (*sy)) * out->x + (j - (*sx))] = this->Re[i * this->x + j];
			out->Im[(i - (*sy)) * out->x + (j - (*sx))] = this->Im[i * this->x + j];
		}
	}


	delete sx;
	delete sy;
}




//角スペクトル法のHを格納、画像の縦横倍の大きさで用意
void My_ComArray_2D::H_kaku(double lam, double z, double d)
{
	double* u, * v, * Retmp, * Imtmp;
	u = new double;
	v = new double;

	Retmp = new double[x * y];
	Imtmp = new double[x * y];

	*u = 1 / ((double)x * d);
	*v = 1 / ((double)y * d);

	//H計算
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			Retmp[i * x + j] = cos(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr((*u) * ((double)j - x / 2)) - sqr((*v) * ((double)i - y / 2))));
			Imtmp[i * x + j] = sin(2 * M_PI * z * sqrt(sqr(1 / lam) - sqr((*u) * ((double)j - x / 2)) - sqr((*v) * ((double)i - y / 2))));
		}
	}

	//Hシフト
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			if (j < x / 2 && i < y / 2) {
				this->Re[i * x + j] = Retmp[(i + y / 2) * x + (j + x / 2)];
				this->Re[(i + y / 2) * x + (j + x / 2)] = Retmp[i * x + j];
				this->Im[i * x + j] = Imtmp[(i + y / 2) * x + (j + x / 2)];
				this->Im[(i + y / 2) * x + (j + x / 2)] = Imtmp[i * x + j];
			}
			else if (j >= x / 2 && i < y / 2) {
				this->Re[i * x + j] = Retmp[(i + y / 2) * x + (j - x / 2)];
				this->Re[(i + y / 2) * x + (j - x / 2)] = Retmp[i * x + j];
				this->Im[i * x + j] = Imtmp[(i + y / 2) * x + (j - x / 2)];
				this->Im[(i + y / 2) * x + (j - x / 2)] = Imtmp[i * x + j];
			}
		}
	}

	delete[]Retmp;
	delete[]Imtmp;
	delete u;
	delete v;
}

//現在格納されいてるHを使って、指定したinの角スペクトル法を実行後、結果をoutに格納
void My_ComArray_2D::kaku(My_ComArray_2D* out, My_ComArray_2D* in)
{
	//Hの大きさで用意
	My_ComArray_2D* inpad;
	inpad = new My_ComArray_2D(this->s, this->x, this->y);

	//in(G)を0埋めして2倍の大きさ
	in->zeropad(inpad);

	//Gをfft
	My_Fft* fft1;
	fft1 = new My_Fft(this->x, this->y);
	fft1->data_to_in(inpad->Re, inpad->Im);
	fft1->fft2d();
	fft1->out_to_data(inpad->Re, inpad->Im);

	//GXHを計算
	inpad->mul_complex(this);


	//GXHをifft
	fft1->data_to_in(inpad->Re, inpad->Im);
	fft1->ifft2d();
	fft1->out_to_data(inpad->Re, inpad->Im);

	delete fft1;

	//0埋めした部分を取り除いて、outに格納
	inpad->extract_center(out);


	delete inpad;
}



//ランダム位相拡散板
void My_ComArray_2D::diffuser_Random(int rand_seed)
{
	srand(rand_seed);
	double random_num;

	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {

			random_num = rand();
			this->Re[i * x + j] = cos(((double)random_num / RAND_MAX) * 2 * M_PI);
			this->Im[i * x + j] = sin(((double)random_num / RAND_MAX) * 2 * M_PI);
		}
	}
}


//--------------------------------------------LENS-------------------------------------------

void My_LensArray::Lens()
{
	if (approx) {
		//近似
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				double dx, dy;
				dx = ((double)j - (x / 2)) * d;
				dy = ((double)i - (y / 2)) * d;

				this->Re[i * x + j] = cos((-2 * M_PI / lamda) * (sqr(dx) + sqr(dy)) / (2 * f));
				this->Im[i * x + j] = sin((-2 * M_PI / lamda) * (sqr(dx) + sqr(dy)) / (2 * f));
			}
		}

	}
	else {
		//近似なし
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				double dx, dy;
				dx = ((double)j - (x / 2)) * d;
				dy = ((double)i - (y / 2)) * d;

				this->Re[i * x + j] = cos((-2 * M_PI / lamda) * sqrt(sqr(dx) + sqr(dy) + sqr(f)));
				this->Im[i * x + j] = sin((-2 * M_PI / lamda) * sqrt(sqr(dx) + sqr(dy) + sqr(f)));
			}
		}

	}
}


void My_LensArray::diffuser_Lensarray(int ls)
{
	
	//微小単一レンズ作成
	My_LensArray* mini_lens;
	mini_lens = new My_LensArray(s, ls, ls, approx, f, lamda, d);
	mini_lens->Lens();


	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			this->Re[i * x + j] = mini_lens->Re[(i % ls) * ls + (j % ls)];
			this->Im[i * x + j] = mini_lens->Im[(i % ls) * ls + (j % ls)];
		}
	}

	//微小単一レンズ削除
	delete mini_lens;

}