// Three-dimensional FFT based on row–column algorithm

subroutine fft3d(x,n1 ,n2 ,n3)
implicit real*8 (a-h,o-z)
complex *16 x(n1,n2 ,n3),y(n2 ,n3,n1),z(n3 ,n1 ,n2)
! Step 1: n2*n3 individual n1 -point multicolumn FFTs
do k=1,n3
do j=1,n2
call fft(x(1,j,k),n1)
end do
end do
! Step 2: Transposition
do i=1,n1
do k=1,n3
do j=1,n2
y(j,k,i)=x(i,j,k)
end do
end do
end do
! Step 3: n3*n1 individual n2 -point multicolumn FFTs
do i=1,n1
do k=1,n3
call fft(y(1,k,i),n2)
end do
end do
! Step 4: Transposition
do j=1,n2
do i=1,n1
do k=1,n3
z(k,i,j)=y(j,k,i)
end do
end do
end do
! Step 5: n1*n2 individual n3 -point multicolumn FFTs
do j=1,n2
do i=1,n1
call fft(z(1,i,j),n3)
end do
end do
! Step 6: Transposition
do k=1,n3
do j=1,n2
do i=1,n1
x(i,j,k)=z(k,i,j)
end do
end do
end do
return
end