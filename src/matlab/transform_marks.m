function marks = transform_marks(marks_raw)
% Log-transforms the raw severity marks and normalizes them to mean one (Eq. 4).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    marks = log(1 + double(marks_raw(:)));
    marks = marks ./ mean(marks);
end
