% problème "Fizz Buzz" procédural en matlab/octave
for i=1:100
if mod(i,3)==0
  if mod(i,5)==0
  printf('fizzbuzz ')
  else
  printf('fizz ',i)
  endif
elseif mod(i,5)==0
    printf('buzz ',i)
else
  printf('%d ',i)
endif
end
printf('\n')



