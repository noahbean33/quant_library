// Two-dimensional FFT based on row–column algorithm

subroutine fft2d(x,n1 ,n2)
implicit real*8 (a-h,o-z)
complex *16 x(n1,n2),y(n2 ,n1)
! Step 1: n2 individual n1 -point multicolumn FFTs
do j=1,n2
call fft(x(1,j),n1)
end do
! Step 2: Transposition
do i=1,n1
do j=1,n2
y(j,i)=x(i,j)
end do
end do
! Step 3: n1 individual n2 -point multicolumn FFTs
do i=1,n1
call fft(y(1,i),n2)
end do
! Step 4: Transposition
do j=1,n2
do i=1,n1
x(i,j)=y(j,i)
end do
end do
return
end